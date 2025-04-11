import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import plotly.express as px
import random
import math

# --- Класс LMSR (можно использовать предыдущий LMSRMarketMakerAdv) ---
# Оставляем класс LMSRMarketMakerAdv из предыдущего ответа, т.к. он уже
# поддерживает продажу и расчеты. Убедимся, что он импортирован или вставлен сюда.
# (Вставил сюда для полноты)
class LMSRMarketMakerAdv:
    def __init__(self, num_outcomes, b, fee_rate=0.02):
        if b <= 0:
            raise ValueError("Параметр ликвидности 'b' должен быть положительным.")
        self.num_outcomes = num_outcomes
        self.b = b
        self.fee_rate = fee_rate
        # q - количество акций каждого исхода в обращении
        self.q = np.zeros(num_outcomes, dtype=float)
        self.total_fees_collected = 0.0
        self.lp_fees_collected = 0.0 # Отдельно для ЛП
        self.total_volume_traded = 0.0 # Общий объем (сумма трат / получ денег до комиссии)

    def cost_function(self, quantities):
        quantities = np.asarray(quantities, dtype=float)
        if self.b <= 0 : return 0.0 # Бесконечная ликвидность / ошибка

        with np.errstate(over='ignore', divide='ignore'): # Подавляем ошибки overflow/zero-div
            scaled_q = quantities / self.b
            # Проверка на бесконечность ДО вычисления max_q_b
            if not np.all(np.isfinite(scaled_q)):
                # Если есть бесконечность, C -> inf, но вернем большое число или обработаем
                 # st.sidebar.warning("Бесконечность в scaled_q, C не определена.")
                 # Можно вернуть очень большое число или как-то иначе обработать
                 # Пока просто используем np.nan_to_num для замены inf
                 scaled_q = np.nan_to_num(scaled_q, nan=0.0, posinf=1e6, neginf=-1e6) # Ограничение

            max_q_b = np.max(scaled_q) if quantities.size > 0 else 0

             # Стабильный расчет: exp(x - max(x)) * exp(max(x))
            sum_exp = np.sum(np.exp(scaled_q - max_q_b))

            if sum_exp <= 0 or not np.isfinite(sum_exp) or not np.isfinite(max_q_b):
                 min_cost = self.b * np.log(self.num_outcomes) if self.num_outcomes > 0 else 0.0
                 return min_cost

            cost = self.b * (max_q_b + np.log(sum_exp))

        # Гарантируем, что C не меньше начального значения b*ln(N)
        min_cost = self.b * np.log(self.num_outcomes) if self.num_outcomes > 0 else 0.0
        # Проверяем на NaN перед max()
        if not np.isfinite(cost):
             return min_cost
        return max(cost, min_cost)


    def get_prices(self):
        if self.b <= 0:
             return np.full(self.num_outcomes, 1.0 / self.num_outcomes)

        with np.errstate(over='ignore', divide='ignore'):
            scaled_q = self.q / self.b
            if not np.all(np.isfinite(scaled_q)):
                scaled_q = np.nan_to_num(scaled_q, nan=0.0, posinf=1e6, neginf=-1e6)

            max_q_b = np.max(scaled_q) if self.q.size > 0 else 0
            exps = np.exp(scaled_q - max_q_b)
            sum_exps = np.sum(exps)

            if sum_exps <= 0 or not np.isfinite(sum_exps):
                 return np.full(self.num_outcomes, 1.0 / self.num_outcomes)

            prices = exps / sum_exps

        # Проверка на NaN и замена на равные вероятности
        prices = np.nan_to_num(prices, nan=(1.0/self.num_outcomes))

        # Убедимся, что сумма = 1 (из-за округлений)
        prices_sum = np.sum(prices)
        if np.isclose(prices_sum, 0):
             return np.full(self.num_outcomes, 1.0 / self.num_outcomes)
        prices /= prices_sum
        return prices

    def _cost_change_for_shares(self, outcome_index, shares_delta, current_q):
        """Рассчитывает ИЗМЕНЕНИЕ стоимости C при добавлении/удалении shares_delta"""
        if np.isclose(shares_delta, 0): return 0.0

        q_old = np.copy(current_q)
        c_old = self.cost_function(q_old)

        q_new = np.copy(q_old)
        q_new[outcome_index] += shares_delta
        # Важно: не допускать отрицательное количество акций
        if q_new[outcome_index] < -1e-9: # Небольшой допуск
             shares_delta = -q_old[outcome_index] # Корректируем дельту
             q_new[outcome_index] = 0.0
             if np.isclose(shares_delta, 0): return 0.0 # Ничего не продали

        c_new = self.cost_function(q_new)

        # Проверка на NaN или Inf в стоимостях
        if not np.isfinite(c_old) or not np.isfinite(c_new):
            # st.sidebar.warning(f"Некорректная стоимость C при расчете изменения: C_old={c_old}, C_new={c_new}")
            return 0.0 # Возвращаем 0, если стоимость невалидна

        cost_diff = c_new - c_old
        return cost_diff


    def calculate_shares_for_amount(self, outcome_index, amount_to_spend_on_shares, current_q):
        """Сколько акций КУПИТЬ за amount_to_spend_on_shares"""
        if amount_to_spend_on_shares <= 1e-9: return 0.0

        # Функция ошибки: CostChange(shares) - TargetCost = 0
        def error_function(shares_to_buy):
            cost_change = self._cost_change_for_shares(outcome_index, shares_to_buy, current_q)
            return cost_change - amount_to_spend_on_shares

        # Границы для поиска
        lower_bound = 0.0
        prices = self.get_prices() # Используем текущие цены объекта
        current_price = prices[outcome_index] if outcome_index < len(prices) else (1.0/self.num_outcomes)

        # Оценка верхней границы
        if current_price < 1e-9:
             # Цена почти 0, потенциально можно купить очень много.
             # Оценка на основе b кажется разумной
             upper_bound_guess = 50 * self.b + amount_to_spend_on_shares # Увеличим запас
        else: upper_bound_guess = (amount_to_spend_on_shares / current_price) * 10 + 1.0 # Запас x10

        try:
            # Важно: нужно проверить сами значения error_function
            val_at_zero = error_function(1e-9)
            if not np.isfinite(val_at_zero): val_at_zero = -amount_to_spend_on_shares # Если ошибка, считаем, что цена норм

            # Проверяем верхнюю границу итеративно
            attempts = 0
            max_attempts = 5
            val_at_upper = error_function(upper_bound_guess)
            while val_at_upper < 0 and attempts < max_attempts:
                 if not np.isfinite(val_at_upper): break # Выход если ошибка расчета
                 upper_bound_guess *= 10
                 val_at_upper = error_function(upper_bound_guess)
                 attempts += 1

            if not np.isfinite(val_at_upper): # Если не удалось найти конечную верхнюю границу
                  # st.sidebar.warning(f"Ошибка расчета error_function на верхней границе {upper_bound_guess}")
                  return 0.0 # Не удалось найти решение

            if val_at_zero >= 0: return 0.0 # Не можем купить
            if val_at_upper < 0: return 0.0 # Не удалось найти верхнюю границу, где f > 0

            # Если знаки разные, запускаем решатель
            shares_bought = brentq(error_function, lower_bound, upper_bound_guess, xtol=1e-7, rtol=1e-7)
            return max(0.0, shares_bought) # Убедимся, что не отрицательно
        except (ValueError, OverflowError) as e:
            # st.sidebar.error(f"Ошибка решателя (brentq) при покупке: {e}")
            # st.sidebar.error(f"Границы: [{lower_bound:.2f}, {upper_bound_guess:.2f}], Знач: [{val_at_zero:.4f}, {val_at_upper:.4f}]")
            # st.sidebar.error(f"q={current_q}, P={prices}")
            return 0.0

    def calculate_amount_for_shares(self, outcome_index, shares_to_sell, current_q):
        """Сколько ДЕНЕГ получить за ПРОДАЖУ shares_to_sell акций"""
        if shares_to_sell <= 1e-9: return 0.0

         # Убедимся, что не продаем больше, чем есть
        shares_available = current_q[outcome_index]
        shares_to_sell_actual = min(shares_to_sell, shares_available)
        if shares_to_sell_actual <= 1e-9: return 0.0

        # Изменение стоимости будет отрицательным, берем модуль
        cost_change = self._cost_change_for_shares(outcome_index, -shares_to_sell_actual, current_q)
        # Cost change < 0 for selling
        amount_received = -cost_change
        return max(0.0, amount_received) # Сумма не может быть отрицательной

    def execute_trade(self, user_id, action, outcome_index, value, user_shares, lp_fee_share=0.0):
        """
        Выполняет сделку (покупку или продажу).
        action: 'buy' или 'sell'
        value: сумма для покупки или количество акций для продажи
        user_shares: ТЕКУЩИЙ баланс акций пользователя (numpy array)
        lp_fee_share: Доля комиссии, идущая LP (e.g., 0.5 для 50%)
        Возвращает словарь с деталями сделки или None при ошибке.
        """
        q_before = np.copy(self.q)
        c_before = self.cost_function(q_before)
        prices_before = self.get_prices()

        fee_paid = 0.0
        shares_delta = 0.0
        money_delta_lmsr = 0.0 # Изменение C (без комиссии)
        money_delta_user = 0.0 # Деньги для пользователя (с уч. комиссии)
        effective_price = 0.0
        gross_trade_value = 0.0 # Сумма до комиссии (для объема)

        if action == 'buy':
            amount_to_spend = value
            if amount_to_spend <= 0: return None
            gross_trade_value = amount_to_spend

            fee_paid = amount_to_spend * self.fee_rate
            cost_for_shares = amount_to_spend * (1 - self.fee_rate)
            money_delta_lmsr = cost_for_shares # C увеличивается на эту сумму
            money_delta_user = -amount_to_spend # Пользователь тратит

            shares_delta = self.calculate_shares_for_amount(outcome_index, cost_for_shares, q_before)

            if shares_delta > 1e-9:
                self.q[outcome_index] += shares_delta
                effective_price = cost_for_shares / shares_delta
            else: return None # Покупка не удалась

        elif action == 'sell':
            shares_to_sell = min(value, user_shares[outcome_index]) # Не продать больше, чем есть
            if shares_to_sell <= 1e-9: return None
            shares_delta = -shares_to_sell # Отрицательное изменение q

            # Рассчитываем, сколько денег из LMSR ДО комиссии
            gross_amount_received = self.calculate_amount_for_shares(outcome_index, shares_to_sell, q_before)
            if gross_amount_received <= 1e-9: return None # Не удалось продать (цена 0?)
            gross_trade_value = gross_amount_received

            fee_paid = gross_amount_received * self.fee_rate
            net_amount_received = gross_amount_received * (1 - self.fee_rate)
            money_delta_lmsr = -gross_amount_received # C уменьшается на сумму до комиссии
            money_delta_user = net_amount_received # Пользователь получает

            self.q[outcome_index] += shares_delta # Вычитаем акции
            effective_price = net_amount_received / shares_to_sell

        else: return None # Неизвестное действие

        # Обновляем общие счетчики
        self.total_volume_traded += gross_trade_value # Объем = сумма до комиссии
        self.total_fees_collected += fee_paid
        self.lp_fees_collected += fee_paid * lp_fee_share

        # Расчет ПОСЛЕ сделки
        c_after = self.cost_function(self.q)
        prices_after = self.get_prices()

        # Проверка изменения C (Опционально, для отладки)
        cost_change_check = c_after - c_before
        if not np.isclose(cost_change_check, money_delta_lmsr, rtol=1e-3, atol=1e-4):
             #pass
             st.sidebar.warning(f"Расхождение C: ΔC={cost_change_check:.4f}, Ожидалось={money_delta_lmsr:.4f}")


        trade_details = {
            "User": user_id,
            "Action": action,
            "Исход": outcome_index,
            "Value": value, # Сумма покупки / кол-во продажи
            "Shares_Δ": shares_delta,
            "Fee ($)": fee_paid,
            "LP_Fee ($)": fee_paid * lp_fee_share,
            "Money_Δ_User ($)": money_delta_user, # Изменение баланса юзера
            "Money_Δ_LMSR ($)": money_delta_lmsr, # Изменение C (без комиссии)
            "Eff_Price ($)": effective_price,
            "P_yes (До)": prices_before[0],
            "P_no (До)": prices_before[1],
            "C (До)": c_before,
            "P_yes (После)": prices_after[0],
            "P_no (После)": prices_after[1],
            "C (После)": c_after,
            "q_yes (После)": self.q[0],
            "q_no (После)": self.q[1],
        }
        return trade_details

# --- Вспомогательные функции ---
def update_avg_price(old_q, old_avg_p, shares_bought, eff_price):
    """Обновляет среднюю цену покупки"""
    if shares_bought <= 1e-9:
        return old_avg_p
    # Если раньше не было акций, новая средняя цена = текущая эффективная
    if old_q < 1e-9:
        return eff_price
    # Иначе взвешенное среднее
    new_q = old_q + shares_bought
    new_avg_p = (old_q * old_avg_p + shares_bought * eff_price) / new_q
    return new_avg_p

# --- Функция Симуляции ---
def run_simulation(params):
    """Запускает симуляцию с заданными параметрами"""
    market = LMSRMarketMakerAdv(num_outcomes=2, b=params['b'], fee_rate=params['fee_rate'])

    # Инициализация пользователей
    users = {}
    num_informed = int(params['num_users'] * params['informed_ratio'])
    informed_indices = random.sample(range(params['num_users']), num_informed)

    for i in range(params['num_users']):
        user_id = f"User_{i}"
        is_informed = i in informed_indices
        # Начальное представление об "истине" (зависит от сценария)
        true_prob_yes = 0.5 # По умолчанию
        if is_informed:
            if params['scenario'] == 'Неопределенный Рынок':
                # Колеблется около 0.5
                true_prob_yes = random.uniform(0.4, 0.6)
            elif params['scenario'] == 'Ранний Тренд':
                # Смещено к истинному исходу (params['final_true_outcome'])
                noise = random.gauss(0, 0.05) # Небольшой шум
                true_prob_yes = np.clip(params['final_true_outcome'] + noise, 0.05, 0.95)
            elif params['scenario'] == 'Постепенный Тренд':
                # Начнем около 0.5
                 true_prob_yes = random.uniform(0.45, 0.55)
            else: # По умолчанию - Неопределенный
                 true_prob_yes = random.uniform(0.4, 0.6)

        users[user_id] = {
            'id': user_id,
            'shares': np.zeros(2, dtype=float),
            'avg_price': np.zeros(2, dtype=float), # Средняя цена покупки
            'is_informed': is_informed,
            'true_prob': true_prob_yes, # Представление информированного
            'cash_balance': 1000.0 # Добавим баланс для реалистичности? Пока нет.
        }

    user_ids = list(users.keys())
    records = []
    progress_bar = st.progress(0, text="Симуляция запущена...")

    for step in range(params['num_steps']):
        user_id = random.choice(user_ids)
        user = users[user_id]
        current_prices = market.get_prices()
        current_market_prob_yes = current_prices[0]
        action = None
        outcome_index = -1
        value = 0.0 # Сумма для покупки / Кол-во для продажи

        # Обновление true_prob для сценария "Постепенный Тренд"
        if user['is_informed'] and params['scenario'] == 'Постепенный Тренд':
            progress_ratio = step / params['num_steps']
            target_prob = params['final_true_outcome']
            start_prob = 0.5 # Условно
            # Линейная интерполяция к финальной "истине"
            user['true_prob'] = np.clip(start_prob + (target_prob - start_prob) * progress_ratio + random.gauss(0, 0.02), 0.01, 0.99)


        # --- Логика принятия решений ---
        consider_selling = params['allow_selling'] and (user['shares'][0] > 1e-6 or user['shares'][1] > 1e-6)
        sell_decision = False

        # --- Логика ПРОДАЖИ (если разрешено) ---
        if consider_selling:
            potential_sells = []
            # Проверяем условия TP/SL для каждой позиции
            for idx in range(2):
                if user['shares'][idx] > 1e-6:
                    avg_p = user['avg_price'][idx]
                    current_p = current_prices[idx]
                    profit_margin = (current_p - avg_p) / avg_p if avg_p > 1e-6 else 0

                    # Take Profit
                    if profit_margin >= params['take_profit_threshold']:
                         if (user['is_informed']) or (not user['is_informed'] and random.random() < 0.1): # Неинф. продают реже
                             potential_sells.append({'action': 'sell', 'outcome': idx, 'reason': 'TP'})
                    # Stop Loss
                    elif profit_margin <= -params['stop_loss_threshold']:
                        if (user['is_informed']) or (not user['is_informed'] and random.random() < 0.2): # Неинф. режут убытки чаще
                            potential_sells.append({'action': 'sell', 'outcome': idx, 'reason': 'SL'})
            # Рандомная продажа для неинформированных
            if not user['is_informed'] and random.random() < 0.02: # Малый шанс продать просто так
                 sellable_idx = -1
                 if user['shares'][0] > 1e-6 and user['shares'][1] > 1e-6: sellable_idx = random.choice([0,1])
                 elif user['shares'][0] > 1e-6: sellable_idx = 0
                 elif user['shares'][1] > 1e-6: sellable_idx = 1
                 if sellable_idx != -1:
                      potential_sells.append({'action': 'sell', 'outcome': sellable_idx, 'reason': 'Random'})

            if potential_sells:
                # Выбираем одну из возможных продаж (можно усложнить - например, приоритет SL)
                chosen_sell = random.choice(potential_sells)
                action = 'sell'
                outcome_index = chosen_sell['outcome']
                # Продаем часть позиции (информированные могут продавать больше)
                sell_fraction = random.uniform(0.1, params['informed_sell_fraction'] if user['is_informed'] else params['uninformed_sell_fraction'])
                value = user['shares'][outcome_index] * sell_fraction
                sell_decision = True

        # --- Логика ПОКУПКИ (если не продали) ---
        if not sell_decision:
            action = 'buy'
            buy_decision_made = False
            if user['is_informed']:
                perceived_prob = user['true_prob']
                # Покупаем, если рыночная цена выгоднее нашего представления
                if current_market_prob_yes < perceived_prob - params['informed_threshold']: # Цена YES ниже "истины" -> покупаем YES
                    outcome_index = 0
                    buy_decision_made = True
                elif current_market_prob_yes > perceived_prob + params['informed_threshold']: # Цена YES выше "истины" -> покупаем NO (цена NO ниже)
                    outcome_index = 1
                    buy_decision_made = True

                if buy_decision_made:
                    # Сумма зависит от уверенности/расхождения цен (можно усложнить)
                    value = random.uniform(params['informed_trade_value_min'], params['informed_trade_value_max'])
                else: # Цена близка к ожидаемой, не торгуем
                    action = None

            else: # Неинформированный пользователь
                if random.random() < params['uninformed_trade_prob']: # Вероятность торговли
                     outcome_index = random.choice([0, 1])
                     value = random.uniform(params['uninformed_trade_value_min'], params['uninformed_trade_value_max'])
                else: # Не торгуем
                     action = None

        # --- Выполнение Сделки ---
        if action:
            trade_details = market.execute_trade(
                user_id,
                action,
                outcome_index,
                value,
                user['shares'], # Передаем текущий баланс юзера
                params['lp_fee_share'] if params['enable_lps'] else 0.0
            )

            if trade_details:
                # Обновляем состояние пользователя
                shares_delta = trade_details['Shares_Δ']
                user['shares'][outcome_index] += shares_delta
                # Обновляем среднюю цену только при ПОКУПКЕ
                if action == 'buy':
                    user['avg_price'][outcome_index] = update_avg_price(
                        user['shares'][outcome_index] - shares_delta, # q до сделки
                        user['avg_price'][outcome_index],
                        shares_delta,
                        trade_details['Eff_Price ($)']
                    )

                # Запись данных
                trade_details["Step"] = step + 1
                trade_details["Total_Volume ($)"] = market.total_volume_traded
                trade_details["Total_Fees ($)"] = market.total_fees_collected
                trade_details["LP_Fees ($)"] = market.lp_fees_collected
                trade_details["User_Informed"] = user['is_informed']
                trade_details["User_True_Prob"] = user['true_prob'] if user['is_informed'] else 0.5
                records.append(trade_details)

        progress_bar.progress((step + 1) / params['num_steps'], text=f"Симуляция: Шаг {step+1}/{params['num_steps']}")

    progress_bar.progress(1.0, text="Симуляция завершена!")
    if records:
        results_df = pd.DataFrame(records)
        # Добавляем расчет остатка платформы и прибыли LP в конце
        final_state = results_df.iloc[-1]
        final_C = final_state['C (После)']
        final_q_yes = final_state['q_yes (После)']
        final_q_no = final_state['q_no (После)']

        platform_remainder_if_yes = final_C - final_q_yes
        platform_remainder_if_no = final_C - final_q_no

        lp_profit_from_fees = final_state['LP_Fees ($)']
        # Прибыль LP от излишка = Доля * max(0, Излишек)
        lp_profit_from_surplus_if_yes = params['lp_surplus_share'] * max(0, platform_remainder_if_yes)
        lp_profit_from_surplus_if_no = params['lp_surplus_share'] * max(0, platform_remainder_if_no)

        final_metrics = {
            "platform_remainder_if_yes": platform_remainder_if_yes,
            "platform_remainder_if_no": platform_remainder_if_no,
            "lp_profit_from_fees": lp_profit_from_fees,
            "lp_profit_from_surplus_if_yes": lp_profit_from_surplus_if_yes,
            "lp_profit_from_surplus_if_no": lp_profit_from_surplus_if_no
        }
        return results_df, final_metrics
    else:
        return None, None


# --- Интерфейс Streamlit ---
st.set_page_config(layout="wide", page_title="LMSR Симулятор v3: Профи")
st.title("⚡ LMSR Симулятор v3: Профессиональный Анализ")
st.markdown("Глубокое погружение в динамику рынка с гибкими настройками поведения трейдеров и модели LP.")

# --- Инициализация Session State ---
default_params = {
    # Рынок
    'b': 1000.0, 'fee_rate': 0.02,
    # Симуляция
    'num_users': 100, 'num_steps': 1000, 'scenario': 'Неопределенный Рынок', 'final_true_outcome': 0.8,
    # Пользователи
    'informed_ratio': 0.2, # 20% информированных
    'informed_trade_value_min': 50.0, 'informed_trade_value_max': 200.0, 'informed_threshold': 0.05, # Отклонение для торговли
    'uninformed_trade_value_min': 5.0, 'uninformed_trade_value_max': 25.0, 'uninformed_trade_prob': 0.1, # Вероятность торговли неинф.
    # Продажа
    'allow_selling': True, 'take_profit_threshold': 0.15, 'stop_loss_threshold': 0.10, # 15% TP, 10% SL
    'informed_sell_fraction': 0.7, 'uninformed_sell_fraction': 0.3, # Доля продажи
    # LP
    'enable_lps': True, 'lp_fee_share': 0.5, 'lp_surplus_share': 0.2 # 50% от комиссий, 20% от излишка
}

# Инициализация state, если его нет
for key, value in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = value
if 'results' not in st.session_state: st.session_state.results = None
if 'final_metrics' not in st.session_state: st.session_state.final_metrics = None

# --- Боковая панель ---
with st.sidebar:
    st.header("⚙️ Настройки Симуляции")

    with st.expander("Рыночные Параметры", expanded=True):
        st.session_state['b'] = st.number_input("Начальная Ликвидность (b)", min_value=1.0, value=st.session_state['b'], step=100.0, format="%.1f", help="Основа ликвидности. Рекомендация: 100-100000 в зависимости от ожидаемого объема.")
        fee_percent = st.number_input("Ставка Комиссии (%)", min_value=0.0, max_value=20.0, value=st.session_state['fee_rate'] * 100, step=0.1, format="%.1f", help="Общая комиссия. Рекомендация: 0.5-3%.")
        st.session_state['fee_rate'] = fee_percent / 100.0

    with st.expander("Основные Параметры Симуляции", expanded=True):
        st.session_state['num_users'] = st.number_input("Количество Пользователей", min_value=2, value=st.session_state['num_users'], step=10, help="Общее число трейдеров.")
        st.session_state['num_steps'] = st.number_input("Количество Шагов (Трейдов)", min_value=10, value=st.session_state['num_steps'], step=100, help="Число итераций симуляции.")
        st.session_state['scenario'] = st.selectbox("Сценарий Рынка", ['Неопределенный Рынок', 'Ранний Тренд', 'Постепенный Тренд'], index=['Неопределенный Рынок', 'Ранний Тренд', 'Постепенный Тренд'].index(st.session_state['scenario']), help="Определяет поведение информированных трейдеров.")
        if st.session_state['scenario'] != 'Неопределенный Рынок':
            st.session_state['final_true_outcome'] = st.number_input("Финальная 'Истинная' P(Да)", min_value=0.01, max_value=0.99, value=st.session_state['final_true_outcome'], step=0.05, format="%.2f", help="К какой вероятности стремится рынок в сценариях с трендом.")

    with st.expander("Параметры Трейдеров"):
        informed_ratio_percent = st.number_input("Доля Информированных Трейдеров (%)", min_value=0, max_value=100, value=int(st.session_state['informed_ratio'] * 100), step=5, help="Процент трейдеров, имеющих представление об 'истинной' вероятности.")
        st.session_state['informed_ratio'] = informed_ratio_percent / 100.0
        st.markdown("--- **Информированные** ---")
        st.session_state['informed_trade_value_min'] = st.number_input("Мин. Сумма Сделки Инф. ($)", min_value=1.0, value=st.session_state['informed_trade_value_min'], step=10.0, format="%.1f")
        st.session_state['informed_trade_value_max'] = st.number_input("Макс. Сумма Сделки Инф. ($)", min_value=1.0, value=st.session_state['informed_trade_value_max'], step=10.0, format="%.1f")
        st.session_state['informed_threshold'] = st.number_input("Порог Отклонения Цены Инф.", min_value=0.0, max_value=0.5, value=st.session_state['informed_threshold'], step=0.01, format="%.2f", help="Насколько рын. цена должна отличаться от 'истинной', чтобы инф. трейдер торговал.")
        st.markdown("--- **Неинформированные** ---")
        st.session_state['uninformed_trade_value_min'] = st.number_input("Мин. Сумма Сделки Неинф. ($)", min_value=0.1, value=st.session_state['uninformed_trade_value_min'], step=1.0, format="%.1f")
        st.session_state['uninformed_trade_value_max'] = st.number_input("Макс. Сумма Сделки Неинф. ($)", min_value=0.1, value=st.session_state['uninformed_trade_value_max'], step=1.0, format="%.1f")
        st.session_state['uninformed_trade_prob'] = st.number_input("Вероятность Торговли Неинф.", min_value=0.0, max_value=1.0, value=st.session_state['uninformed_trade_prob'], step=0.01, format="%.2f", help="Как часто неинф. трейдер совершает случайную сделку.")

    with st.expander("Параметры Продажи"):
        st.session_state['allow_selling'] = st.toggle("Разрешить Продажу Акций", value=st.session_state['allow_selling'])
        if st.session_state['allow_selling']:
            tp_percent = st.number_input("Порог Take Profit (%)", min_value=1, max_value=200, value=int(st.session_state['take_profit_threshold']*100), step=1, help="При каком % прибыли продавать.")
            sl_percent = st.number_input("Порог Stop Loss (%)", min_value=1, max_value=100, value=int(st.session_state['stop_loss_threshold']*100), step=1, help="При каком % убытка продавать.")
            st.session_state['take_profit_threshold'] = tp_percent / 100.0
            st.session_state['stop_loss_threshold'] = sl_percent / 100.0
            inf_sell_frac_percent = st.number_input("Доля Продажи Инф. (%)", min_value=1, max_value=100, value=int(st.session_state['informed_sell_fraction']*100), step=5, help="Какую часть позиции продает информированный.")
            uninf_sell_frac_percent = st.number_input("Доля Продажи Неинф. (%)", min_value=1, max_value=100, value=int(st.session_state['uninformed_sell_fraction']*100), step=5, help="Какую часть позиции продает неинформированный.")
            st.session_state['informed_sell_fraction'] = inf_sell_frac_percent / 100.0
            st.session_state['uninformed_sell_fraction'] = uninf_sell_frac_percent / 100.0

    with st.expander("Параметры Поставщиков Ликвидности (LP)"):
        st.session_state['enable_lps'] = st.toggle("Включить Модель LP", value=st.session_state['enable_lps'])
        if st.session_state['enable_lps']:
            lp_fee_share_percent = st.number_input("Доля LP в Комиссиях (%)", min_value=0, max_value=100, value=int(st.session_state['lp_fee_share'] * 100), step=5, help="Какой % от ТОРГОВЫХ комиссий идет LP.")
            lp_surplus_share_percent = st.number_input("Доля LP в Излишке Рынка (%)", min_value=0, max_value=100, value=int(st.session_state['lp_surplus_share'] * 100), step=5, help="Какой % от ОСТАТКА средств в LMSR (C - Выплата) после завершения рынка идет LP.")
            st.session_state['lp_fee_share'] = lp_fee_share_percent / 100.0
            st.session_state['lp_surplus_share'] = lp_surplus_share_percent / 100.0

    # Кнопка Запуска
    run_button = st.button("🚀 Запустить Симуляцию", use_container_width=True)

# --- Логика Запуска и Отображения ---
if run_button:
    # Собираем все параметры из state в один словарь
    current_sim_params = {key: st.session_state[key] for key in default_params}
    with st.spinner("Выполняется симуляция... Это может занять некоторое время."):
        st.session_state.results, st.session_state.final_metrics = run_simulation(current_sim_params)
    if st.session_state.results is None:
        st.error("Симуляция не дала результатов. Проверьте параметры.")

# --- Отображение Результатов ---
if st.session_state.results is not None:
    st.header("📈 Результаты Симуляции")
    results_df = st.session_state.results
    final_metrics = st.session_state.final_metrics
    last_state = results_df.iloc[-1]

    # --- Сводка и Ключевые Метрики ---
    st.subheader("🏁 Финальное Состояние и Сводка")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Цена 'ДА'", f"{last_state['P_yes (После)']:.4f}")
        st.metric("Цена 'НЕТ'", f"{last_state['P_no (После)']:.4f}")
        st.metric("Финальное C ($)", f"{last_state['C (После)']:.2f}")
    with col2:
        st.metric("Общий Объем ($)", f"{last_state['Total_Volume ($)']:.2f}")
        st.metric("Всего Комиссий ($)", f"{last_state['Total_Fees ($)']:.2f}")
        platform_fee = last_state['Total_Fees ($)'] - final_metrics['lp_profit_from_fees']
        st.metric("Комиссии Платформы ($)", f"{platform_fee:.2f}", help="Общие комиссии минус доля LP от комиссий.")
    with col3:
         st.metric("Финальное q(Да)", f"{last_state['q_yes (После)']:.2f}")
         st.metric("Финальное q(Нет)", f"{last_state['q_no (После)']:.2f}")
         st.metric("Комиссии LP (от сборов) ($)", f"{final_metrics['lp_profit_from_fees']:.2f}" if st.session_state['enable_lps'] else "N/A")


    # --- Расчет Прибыли Платформы и LP ---
    st.subheader("💰 Анализ Прибыли (Пост-Симуляция)")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Если исход = 'ДА'**")
        rem_yes = final_metrics['platform_remainder_if_yes']
        lp_surplus_yes = final_metrics['lp_profit_from_surplus_if_yes']
        platform_profit_yes = platform_fee + (rem_yes - lp_surplus_yes if rem_yes > 0 else rem_yes) # Платформа получает остаток излишка
        st.metric("Остаток Платформы ($)", f"{rem_yes:.2f}", help="C - q(Да). Положительное значение - излишек.")
        if st.session_state['enable_lps']:
             st.metric("Прибыль LP (Излишек) ($)", f"{lp_surplus_yes:.2f}", help=f"{st.session_state['lp_surplus_share']*100:.0f}% от положительного остатка")
             st.metric("ИТОГО LP ($)", f"{final_metrics['lp_profit_from_fees'] + lp_surplus_yes:.2f}", help="Комиссии + Доля излишка")
             st.metric("ИТОГО Платформа ($)", f"{platform_profit_yes:.2f}", help="Ее доля комиссий + остаток излишка")
        else:
             st.metric("ИТОГО Платформа ($)", f"{platform_fee + rem_yes:.2f}", help="Комиссии + Весь остаток")


    with col_p2:
        st.markdown("**Если исход = 'НЕТ'**")
        rem_no = final_metrics['platform_remainder_if_no']
        lp_surplus_no = final_metrics['lp_profit_from_surplus_if_no']
        platform_profit_no = platform_fee + (rem_no - lp_surplus_no if rem_no > 0 else rem_no)
        st.metric("Остаток Платформы ($)", f"{rem_no:.2f}", help="C - q(Нет). Положительное значение - излишек.")
        if st.session_state['enable_lps']:
             st.metric("Прибыль LP (Излишек) ($)", f"{lp_surplus_no:.2f}", help=f"{st.session_state['lp_surplus_share']*100:.0f}% от положительного остатка")
             st.metric("ИТОГО LP ($)", f"{final_metrics['lp_profit_from_fees'] + lp_surplus_no:.2f}", help="Комиссии + Доля излишка")
             st.metric("ИТОГО Платформа ($)", f"{platform_profit_no:.2f}", help="Ее доля комиссий + остаток излишка")
        else:
            st.metric("ИТОГО Платформа ($)", f"{platform_fee + rem_no:.2f}", help="Комиссии + Весь остаток")


    # --- Графики ---
    st.subheader("📊 Динамика Во Времени")
    tab1, tab2, tab3 = st.tabs(["Цены и Объем", "Комиссии и Прибыль LP", "Стоимость C и Выплаты"])

    with tab1:
        # 1. Цены
        fig_prices = px.line(results_df, x="Step", y=["P_yes (После)", "P_no (После)"], title="Динамика Цен Исходов")
        fig_prices.update_layout(yaxis_range=[0,1], yaxis_title="Цена (Вероятность)", xaxis_title="Шаг Симуляции", legend_title="Исход")
        fig_prices.data[0].name = 'Цена ДА'; fig_prices.data[1].name = 'Цена НЕТ'
        st.plotly_chart(fig_prices, use_container_width=True)
        # 2. Объем
        fig_volume = px.line(results_df, x="Step", y="Total_Volume ($)", title="Накопленный Объем Торгов")
        fig_volume.update_layout(yaxis_title="Общий Объем ($)", xaxis_title="Шаг Симуляции")
        st.plotly_chart(fig_volume, use_container_width=True)

    with tab2:
        # 3. Комиссии (Общие и LP)
        if st.session_state['enable_lps']:
            fig_fees = px.line(results_df, x="Step", y=["Total_Fees ($)", "LP_Fees ($)"], title="Накопленные Комиссии (Общие и Доля LP)")
            fig_fees.data[0].name = 'Всего Комиссий'; fig_fees.data[1].name = 'Комиссии LP (от сборов)'
        else:
            fig_fees = px.line(results_df, x="Step", y=["Total_Fees ($)"], title="Накопленные Общие Комиссии")
            fig_fees.data[0].name = 'Всего Комиссий'
        fig_fees.update_layout(yaxis_title="Сумма ($)", xaxis_title="Шаг Симуляции", legend_title="Тип Комиссии")
        st.plotly_chart(fig_fees, use_container_width=True)

    with tab3:
        # 4. Стоимость C и Макс Выплата
        results_df['Max_Payout'] = results_df[['q_yes (После)', 'q_no (После)']].max(axis=1)
        fig_c_payout = px.line(results_df, x="Step", y=["C (После)", "Max_Payout"], title="Стоимость Системы C vs Макс. Возможная Выплата")
        fig_c_payout.update_layout(yaxis_title="Сумма ($)", xaxis_title="Шаг Симуляции", legend_title="Метрика")
        fig_c_payout.data[0].name = 'Стоимость C'; fig_c_payout.data[1].name = 'Макс. Выплата'
        st.plotly_chart(fig_c_payout, use_container_width=True)

    # --- Таблица Трейдов (опционально) ---
    with st.expander("Показать Детальную Историю Трейдов (Последние 200)"):
        display_cols = ["Step", "User", "User_Informed", "Action", "Исход", "Value", "Shares_Δ", "Fee ($)", "LP_Fee ($)", "Eff_Price ($)", "P_yes (После)", "P_no (После)", "C (После)"]
        display_df_trades = results_df[display_cols].tail(200).copy()
        display_df_trades["Исход"] = display_df_trades["Исход"].map({0: "ДА", 1: "НЕТ"})
        # Форматирование
        for col in display_df_trades.select_dtypes(include=['float64']).columns:
             if col in ["P_yes (После)", "P_no (После)", "Eff_Price ($)"]: display_df_trades[col] = display_df_trades[col].map('{:.4f}'.format)
             elif col == "Shares_Δ": display_df_trades[col] = display_df_trades[col].map('{:.3f}'.format)
             elif col != "Value": display_df_trades[col] = display_df_trades[col].map('{:.2f}'.format)
             # Value может быть суммой или количеством, оставим как есть или форматируем отдельно
             if col == "Value": display_df_trades[col] = display_df_trades[col].map('{:.2f}'.format)

        st.dataframe(display_df_trades, use_container_width=True, height=400)

elif 'results' in st.session_state and st.session_state.results is None and run_button: # Если запускали, но нет рез-тов
     st.error("Симуляция завершилась без успешных сделок. Проверьте параметры, возможно, суммы сделок слишком малы или ликвидность недостаточна.")
else:
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Запустить Симуляцию'.")