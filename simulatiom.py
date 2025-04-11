import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext")

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import plotly.express as px

# --- Класс LMSR Маркет-Мейкера (из предыдущего примера) ---
class LMSRMarketMaker:
    def __init__(self, num_outcomes, b, fee_rate=0.02):
        if b <= 0:
            raise ValueError("Параметр ликвидности 'b' должен быть положительным.")
        self.num_outcomes = num_outcomes
        self.b = b
        self.fee_rate = fee_rate
        self.q = np.zeros(num_outcomes) # Используем numpy массив для q
        self.total_fees_collected = 0.0

    def cost_function(self, quantities):
        # Используем numpy для эффективности и обработки массивов
        quantities = np.asarray(quantities)
        # Трюк для численной стабильности при больших значениях q/b
        max_q_b = np.max(quantities / self.b) if quantities.size > 0 and self.b > 0 else 0
        sum_exp = np.sum(np.exp(quantities / self.b - max_q_b))
        if sum_exp <= 0 or self.b <= 0:
             # Начальное состояние C = b * ln(N)
             return self.b * np.log(self.num_outcomes)
        cost = self.b * (max_q_b + np.log(sum_exp))
        # Возвращаем b*ln(N) если расчетная стоимость меньше (из-за приближений при q~0)
        return max(cost, self.b * np.log(self.num_outcomes))

    def get_prices(self):
        if self.b <= 0:
             return np.full(self.num_outcomes, 1.0 / self.num_outcomes)
        # Трюк для численной стабильности
        max_q_b = np.max(self.q / self.b)
        exps = np.exp(self.q / self.b - max_q_b)
        sum_exps = np.sum(exps)
        if sum_exps <= 0:
             return np.full(self.num_outcomes, 1.0 / self.num_outcomes)
        prices = exps / sum_exps
        # Проверка на NaN из-за возможных 0/0 или inf/inf
        if np.isnan(prices).any():
             return np.full(self.num_outcomes, 1.0 / self.num_outcomes)
        return prices

    def _cost_of_buying_shares(self, outcome_index, shares_to_buy, current_q):
        """Рассчитывает стоимость покупки ТОЧНОГО количества акций ОТНОСИТЕЛЬНО current_q"""
        if shares_to_buy <= 0:
            return 0.0

        q_old = np.copy(current_q)
        c_old = self.cost_function(q_old)

        q_new = np.copy(q_old)
        q_new[outcome_index] += shares_to_buy
        c_new = self.cost_function(q_new)

        # Убедимся, что стоимость не отрицательна (из-за погрешностей)
        cost_diff = c_new - c_old
        return max(0.0, cost_diff)


    def calculate_shares_for_amount(self, outcome_index, amount_to_spend_on_shares, current_q):
        """Находит, сколько акций можно купить за amount_to_spend_on_shares"""
        if amount_to_spend_on_shares <= 1e-9:
            return 0.0

        # Функция ошибки: разница между реальной стоимостью и целевой
        def error_function(shares_to_buy):
            # Передаем текущее состояние q явно
            cost = self._cost_of_buying_shares(outcome_index, shares_to_buy, current_q)
            return cost - amount_to_spend_on_shares

        # --- Определение границ для brentq ---
        lower_bound = 0.0
        # Оценка верхней границы: удвоенное количество акций по текущей цене + небольшой запас
        # Это грубая оценка, может потребовать корректировки в сложных случаях
        prices = self.get_prices() # Используем цены из текущего состояния объекта
        current_price = prices[outcome_index]
        if current_price < 1e-9: # Если цена почти ноль
             # Пытаемся купить очень много акций, но ограничиваем сверху
             # Нужно, чтобы C_new был хотя бы немного больше C_old
             # Очень сложно оценить, поставим большое число
             upper_bound_guess = 10 * self.b # Произвольно, но связано с b
        else:
             # Увеличиваем оценку, чтобы гарантированно покрыть цель
             upper_bound_guess = (amount_to_spend_on_shares / current_price) * 5 + 1.0

        # Проверка знаков на границах интервала для brentq
        try:
            val_at_zero = error_function(1e-9) # Почти ноль
            val_at_upper = error_function(upper_bound_guess)

            # st.write(f"DEBUG: error_function(1e-9) = {val_at_zero}") # Отладка
            # st.write(f"DEBUG: error_function({upper_bound_guess}) = {val_at_upper}") # Отладка

            # Brentq требует f(a) * f(b) < 0
            if val_at_zero >= 0:
                # Не можем купить даже минимальное кол-во (цена уже 1 или сумма слишком мала)
                # st.warning(f"Невозможно купить акции: цена слишком высока или сумма мала.")
                return 0.0
            if val_at_upper < 0:
                # Верхняя граница слишком низкая, нужно увеличить
                upper_bound_guess *= 10
                val_at_upper = error_function(upper_bound_guess)
                # st.write(f"DEBUG: New upper bound guess: {upper_bound_guess}, val_at_upper={val_at_upper}")
                if val_at_upper < 0:
                     # Если и это не помогло, что-то странное
                     # st.warning(f"Не удалось найти подходящую верхнюю границу для решателя.")
                     # Возвращаем оценку, хотя она не точна
                     return upper_bound_guess / 10 # Возвращаем предыдущую границу как максимум
            if np.isclose(val_at_zero, val_at_upper):
                 # Значения слишком близки, решатель может не сработать
                 # st.warning("Значения ошибки на границах близки. Возможна неточность.")
                 # Попробуем вернуть простую оценку
                 if current_price > 1e-9:
                      return amount_to_spend_on_shares / current_price
                 else: return 0.0


            # Используем численный метод для поиска корня
            shares_bought = brentq(error_function, lower_bound, upper_bound_guess, xtol=1e-6, rtol=1e-6)
            return shares_bought

        except ValueError as e:
            st.error(f"Ошибка численного решателя (brentq): {e}. Попробуйте изменить сумму или параметр 'b'.")
            st.error(f"Параметры вызова: outcome={outcome_index}, amount={amount_to_spend_on_shares}")
            st.error(f"Состояние q={current_q}, Цены={self.get_prices()}")
            st.error(f"Границы: [{lower_bound}, {upper_bound_guess}], Значения ошибки: [{val_at_zero}, {val_at_upper}]")
            return 0.0 # Возвращаем 0 в случае ошибки решателя


    def buy_shares(self, outcome_index, amount_to_spend):
        """Симулирует покупку, обновляет состояние и возвращает детали сделки"""
        if amount_to_spend <= 0:
            return None # Ничего не делаем

        # Запоминаем состояние ДО сделки
        q_before = np.copy(self.q)
        c_before = self.cost_function(q_before)
        prices_before = self.get_prices()

        # Расчет комиссии и суммы на акции
        fee_paid = amount_to_spend * self.fee_rate
        cost_for_shares = amount_to_spend * (1 - self.fee_rate)

        # Рассчитываем количество акций
        # Важно: передаем q_before в функцию расчета!
        shares_bought = self.calculate_shares_for_amount(outcome_index, cost_for_shares, q_before)

        if shares_bought > 1e-9:
            # Обновляем состояние маркет-мейкера
            self.q[outcome_index] += shares_bought
            self.total_fees_collected += fee_paid

            # Расчет ПОСЛЕ сделки
            c_after = self.cost_function(self.q)
            prices_after = self.get_prices()

            # Проверка: C_after - C_before должно быть примерно равно cost_for_shares
            cost_check = c_after - c_before
            if not np.isclose(cost_check, cost_for_shares, rtol=1e-4):
                 st.warning(f"Небольшое расхождение в стоимости C: Расчетное ΔC={cost_for_shares:.6f}, Фактическое C_after-C_before={cost_check:.6f}")


            effective_price = cost_for_shares / shares_bought if shares_bought > 0 else 0
            potential_winnings = shares_bought # Выигрыш = $1 за акцию

            trade_details = {
                "Исход": outcome_index,
                "Сумма Трат ($)": amount_to_spend,
                "Комиссия ($)": fee_paid,
                "Стоим. Акций ($)": cost_for_shares,
                "Куплено Акций (Δq)": shares_bought,
                "Эфф. Цена ($)": effective_price,
                "Выигрыш Юзера ($)": potential_winnings,
                "P_yes (До)": prices_before[0], # Индекс 0 = YES
                "P_no (До)": prices_before[1],  # Индекс 1 = NO
                "C (До)": c_before,
                "P_yes (После)": prices_after[0],
                "P_no (После)": prices_after[1],
                "C (После)": c_after,
                "q_yes (После)": self.q[0],
                "q_no (После)": self.q[1],
                "Сумма Комисс ($)": self.total_fees_collected # Накопленная
            }
            return trade_details
        else:
            st.warning(f"Покупка на сумму {amount_to_spend:.2f} не удалась (возможно, цена близка к 1 или сумма слишком мала).")
            return None # Покупка не состоялась

# --- Настройка Streamlit ---
st.set_page_config(layout="wide", page_title="Симулятор LMSR")
st.title("📊 Симулятор рынка предсказаний LMSR")
st.markdown("Интерактивная модель для анализа Logarithmic Market Scoring Rule.")

# --- Инициализация Session State ---
# Используем session_state для сохранения данных между действиями пользователя
if 'market' not in st.session_state:
    st.session_state.b = 100.0
    st.session_state.fee_rate = 0.02
    st.session_state.market = LMSRMarketMaker(num_outcomes=2, b=st.session_state.b, fee_rate=st.session_state.fee_rate)
    st.session_state.trade_history = [] # Список словарей с деталями сделок
    st.session_state.trade_counter = 0

# --- Боковая панель для ввода ---
st.sidebar.header("Параметры Рынка")

# Ввод параметров с обновлением объекта market при изменении
b_input = st.sidebar.number_input(
    "Параметр Ликвидности (b)",
    min_value=1.0,
    value=st.session_state.b,
    step=10.0,
    help="Определяет 'глубину' рынка. Чем выше 'b', тем меньше проскальзывание цен при сделках."
)
fee_rate_input = st.sidebar.slider(
    "Ставка Комиссии (%)",
    min_value=0.0,
    max_value=10.0,
    value=st.session_state.fee_rate * 100,
    step=0.1,
    format="%.1f%%",
    help="Процент от суммы сделки, идущий платформе."
) / 100.0 # Делим на 100 для получения доли

# Кнопка для применения новых параметров и сброса симуляции
if st.sidebar.button("Применить параметры и Сбросить симуляцию", key="apply_reset"):
    st.session_state.b = b_input
    st.session_state.fee_rate = fee_rate_input
    st.session_state.market = LMSRMarketMaker(num_outcomes=2, b=st.session_state.b, fee_rate=st.session_state.fee_rate)
    st.session_state.trade_history = []
    st.session_state.trade_counter = 0
    st.experimental_rerun() # Перезапускаем скрипт для чистого старта

# Обновляем параметры в объекте market, если они изменились
# (но без полного сброса, если кнопка не нажата)
if st.session_state.market.b != b_input or st.session_state.market.fee_rate != fee_rate_input:
     # Делаем это аккуратно, чтобы не потерять q и total_fees при простом изменении слайдера
     st.session_state.market.b = b_input
     st.session_state.market.fee_rate = fee_rate_input
     st.session_state.b = b_input # Обновляем и в стейте для консистентности
     st.session_state.fee_rate = fee_rate_input


st.sidebar.header("Совершить Сделку")
outcome_options = {"ДА (YES)": 0, "НЕТ (NO)": 1}
selected_outcome_name = st.sidebar.selectbox(
    "Выберите исход для покупки:",
    options=list(outcome_options.keys())
)
selected_outcome_index = outcome_options[selected_outcome_name]

amount_input = st.sidebar.number_input(
    "Сумма для покупки ($):",
    min_value=0.01,
    value=10.0,
    step=1.0,
    format="%.2f"
)

trade_button = st.sidebar.button("📈 Выполнить Трейд")

# --- Обработка Трейда ---
if trade_button:
    # Передаем текущее состояние q в market maker для расчета цен до сделки
    st.session_state.market.q = st.session_state.market.q # Просто для ясности
    st.session_state.market.total_fees_collected = st.session_state.market.total_fees_collected # -//-

    # Выполняем покупку
    trade_result = st.session_state.market.buy_shares(selected_outcome_index, amount_input)

    if trade_result:
        # Добавляем номер трейда и сохраняем в историю
        st.session_state.trade_counter += 1
        trade_result["Трейд #"] = st.session_state.trade_counter
        st.session_state.trade_history.append(trade_result)
        st.success(f"Трейд #{st.session_state.trade_counter} успешно выполнен!")
    else:
         # Сообщение об ошибке или предупреждение уже выведено внутри buy_shares
         pass

# --- Отображение Результатов ---

st.header("Текущее Состояние Рынка")

# Получаем текущие данные из объекта market в session_state
current_prices = st.session_state.market.get_prices()
current_q = st.session_state.market.q
current_c = st.session_state.market.cost_function(current_q)
total_fees = st.session_state.market.total_fees_collected

col1, col2, col3, col4 = st.columns(4)
col1.metric("Цена 'ДА'", f"{current_prices[0]:.4f}")
col2.metric("Цена 'НЕТ'", f"{current_prices[1]:.4f}")
col3.metric("Стоимость Системы C($)", f"{current_c:.4f}", help="Текущее значение функции затрат LMSR")
col4.metric("Всего Комиссий ($)", f"{total_fees:.4f}", help="Общая сумма комиссий, собранная платформой")

st.markdown(f"**Количество акций в обращении:** q_yes = `{current_q[0]:.4f}`, q_no = `{current_q[1]:.4f}`")
st.markdown(f"**Макс. выплата (если 'ДА'):** ${current_q[0]:.4f}$")
st.markdown(f"**Макс. выплата (если 'НЕТ'):** ${current_q[1]:.4f}$")
# Проверка самофинансирования (C должно покрывать макс. выплату)
max_payout = max(current_q[0], current_q[1])
if current_c + 1e-5 >= max_payout: # Добавляем допуск на погрешность
    st.success(f"✔️ Проверка покрытия: C ({current_c:.4f}) >= Макс. выплата ({max_payout:.4f})")
else:
    st.error(f"❌ ОШИБКА ПОКРЫТИЯ: C ({current_c:.4f}) < Макс. выплата ({max_payout:.4f})!")


st.header("История Транзакций")

if st.session_state.trade_history:
    # Преобразуем историю в DataFrame для удобного отображения
    history_df = pd.DataFrame(st.session_state.trade_history)

    # Форматирование для лучшей читаемости
    display_df = history_df[[
        "Трейд #", "Исход", "Сумма Трат ($)", "Комиссия ($)", "Стоим. Акций ($)",
        "Куплено Акций (Δq)", "Эфф. Цена ($)", "Выигрыш Юзера ($)",
        "P_yes (До)", "P_no (До)", "C (До)",
        "P_yes (После)", "P_no (После)", "C (После)",
         "Сумма Комисс ($)"
    ]].copy() # Выбираем и копируем нужные столбцы

    # Заменяем индексы исходов на имена
    display_df["Исход"] = display_df["Исход"].map({0: "ДА", 1: "НЕТ"})

    # Форматируем числа
    float_cols = display_df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        if "Цена" in col or "P_" in col:
            display_df[col] = display_df[col].map('{:.4f}'.format)
        elif col in ["Куплено Акций (Δq)", "Выигрыш Юзера ($)"]:
             display_df[col] = display_df[col].map('{:.4f}'.format)
        else:
            display_df[col] = display_df[col].map('{:.2f}'.format)

    st.dataframe(display_df, use_container_width=True)

    # --- Графики ---
    st.header("Визуализация")

    # 1. График изменения цен
    price_history_df = history_df[["Трейд #", "P_yes (После)", "P_no (После)"]].copy()
    price_history_df.rename(columns={"P_yes (После)": "Цена ДА", "P_no (После)": "Цена НЕТ"}, inplace=True)
    # Добавляем начальную точку (Трейд #0)
    initial_state = pd.DataFrame([{"Трейд #": 0, "Цена ДА": 0.5, "Цена НЕТ": 0.5}])
    price_history_df = pd.concat([initial_state, price_history_df], ignore_index=True)

    st.subheader("Динамика Цен Исходов")
    fig_prices = px.line(price_history_df, x="Трейд #", y=["Цена ДА", "Цена НЕТ"],
                         title="Изменение Цен ('Вероятностей') Исходов После Каждого Трейда",
                         markers=True)
    fig_prices.update_layout(yaxis_range=[0,1], yaxis_title="Цена (Вероятность)", xaxis_title="Номер Трейда")
    st.plotly_chart(fig_prices, use_container_width=True)

    # 2. График накопления комиссий
    fees_history_df = history_df[["Трейд #", "Сумма Комисс ($)"]].copy()
    # Добавляем начальную точку (Трейд #0)
    initial_fees = pd.DataFrame([{"Трейд #": 0, "Сумма Комисс ($)": 0.0}])
    fees_history_df = pd.concat([initial_fees, fees_history_df], ignore_index=True)

    st.subheader("Накопленная Сумма Комиссий")
    fig_fees = px.line(fees_history_df, x="Трейд #", y="Сумма Комисс ($)",
                       title="Общая Сумма Комиссий Платформы После Каждого Трейда",
                       markers=True)
    fig_fees.update_layout(yaxis_title="Сумма Комиссий ($)", xaxis_title="Номер Трейда")
    st.plotly_chart(fig_fees, use_container_width=True)

    # 3. График изменения C
    c_history_df = history_df[["Трейд #", "C (После)"]].copy()
    # Добавляем начальную точку (Трейд #0)
    initial_c = pd.DataFrame([{"Трейд #": 0, "C (После)": st.session_state.market.cost_function(np.zeros(2))}]) # Начальное C = b*ln(2)
    c_history_df = pd.concat([initial_c, c_history_df], ignore_index=True)
    c_history_df.rename(columns={"C (После)": "Стоимость Системы C ($)"}, inplace=True)

    st.subheader("Изменение Стоимости Системы C")
    fig_c = px.line(c_history_df, x="Трейд #", y="Стоимость Системы C ($)",
                       title="Значение Функции Затрат C После Каждого Трейда",
                       markers=True)
    fig_c.update_layout(yaxis_title="C ($)", xaxis_title="Номер Трейда")
    st.plotly_chart(fig_c, use_container_width=True)


else:
    st.info("Пока не было совершено ни одной транзакции.")

# --- Дополнительная информация ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Как это работает:**")
st.sidebar.caption("""
LMSR (Logarithmic Market Scoring Rule) - это автоматизированный маркет-мейкер.
- **Параметр 'b'**: Контролирует ликвидность. Выше 'b' = меньше влияние сделок на цену.
- **Цены**: Отражают текущую рыночную оценку вероятности исходов. Сумма цен всегда равна 1.
- **Покупка**: Увеличивает цену купленного исхода и уменьшает цену других. Вы платите 'эффективную' цену, которая выше начальной из-за 'проскальзывания'.
- **Стоимость C**: Внутренний 'счет' системы. Увеличивается на 'Стоимость Акций ($)' при каждой покупке. Гарантирует, что у системы хватит средств на выплаты победителям.
- **Комиссия**: Небольшой процент от 'Суммы Трат', который является доходом платформы. Не влияет на внутреннюю механику LMSR.
""")