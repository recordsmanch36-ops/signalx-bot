import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binance.client import Client
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)

warnings.filterwarnings('ignore')

# ==================== НАСТРОЙКИ ====================
TELEGRAM_BOT_TOKEN = "8835644935:AAHVh6DidbRlyN1byuWRuHIdO8Dqi9K_LTw"
ADMIN_ID = 8775854272  # Твой Telegram ID
REFERRAL_LINK = "https://www.bybit.com/invite?ref=0OKBQEA&medium=referral&utm_campaign=evergreen"

MIN_ADX_FOR_TREND = 25

# База данных зарегистрированных пользователей
REGISTERED_USERS = set()

# Список криптовалют
CRYPTO_LIST = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'ADA': 'ADAUSDT', 'XRP': 'XRPUSDT',
    'SOL': 'SOLUSDT', 'DOGE': 'DOGEUSDT', 'LINK': 'LINKUSDT', 'AAVE': 'AAVEUSDT', 
    'ORDI': 'ORDIUSDT', 'BNB': 'BNBUSDT', 'SUI': 'SUIUSDT', 'TON': 'TONUSDT',
    'DOT': 'DOTUSDT', 'AVAX': 'AVAXUSDT', 'SHIB': 'SHIBUSDT', 'ATOM': 'ATOMUSDT',
    'ARB': 'ARBUSDT'
}

TIMEFRAMES = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY
}

binance_client = Client()

# ==================== РАСЧЕТЫ И ИНДИКАТОРЫ ====================

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2):
    sma = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, sma, lower_band

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.rolling(period).mean()
    
    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def get_historical_data(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    try:
        klines = binance_client.get_klines(symbol=CRYPTO_LIST[symbol], interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception:
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['RSI'] = calculate_rsi(df['close'], 14)
    macd, signal, histogram = calculate_macd(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = histogram
    
    upper_band, middle_band, lower_band = calculate_bollinger_bands(df['close'])
    df['BB_Upper'] = upper_band
    df['BB_Middle'] = middle_band
    df['BB_Lower'] = lower_band
    
    adx, plus_di, minus_di = calculate_adx(df['high'], df['low'], df['close'])
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    return df

def determine_market_direction(df: pd.DataFrame) -> str:
    if df.empty or 'SMA_20' not in df.columns or 'SMA_50' not in df.columns:
        return "NEUTRAL"
    last_row = df.iloc[-1]
    return "BULLISH" if last_row['SMA_20'] > last_row['SMA_50'] else "BEARISH"

# ==================== ТЕКСТОВЫЕ БЛОКИ ====================

def build_analysis_report(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    last_row = df.iloc[-1]
    current_price = last_row['close']
    direction = determine_market_direction(df)
    
    text = "══════════════════════════════════════════\n"
    text += f"📊 **АНАЛИЗ {symbol}**\n"
    text += f"⏱ **Таймфрейм:** {timeframe}\n"
    text += f"💰 **Текущая цена:** `{current_price:.4f}`\n"
    text += "══════════════════════════════════════════\n\n"
    
    text += f"📈 **ТЕНДЕНЦИЯ:** {'📈 Бычья' if direction == 'BULLISH' else '📉 Медвежья'}\n"
    text += f"• **SMA 20:** `{last_row['SMA_20']:.4f}`\n"
    text += f"• **SMA 50:** `{last_row['SMA_50']:.4f}`\n"
    diff = abs(last_row['SMA_20'] - last_row['SMA_50'])
    direction_str = 'Вверх' if last_row['SMA_20'] > last_row['SMA_50'] else 'Вниз'
    text += f"• **Разница:** `{diff:.4f}` ({direction_str})\n\n"
    
    text += f"📊 **ВОЛАТИЛЬНОСТЬ (ATR 14):** `{last_row['ATR']:.4f}`\n\n"
    
    rsi_val = last_row['RSI']
    rsi_status = "Перекупленность (>70)" if rsi_val > 70 else "Перепроданность (<30)" if rsi_val < 30 else "Нейтральный"
    text += f"📉 **RSI (14):** `{rsi_val:.2f}` — {rsi_status}\n"
    
    macd_diff = last_row['MACD'] - last_row['MACD_Signal']
    macd_sig = "Бычий (MACD > Signal)" if macd_diff > 0 else "Медвежий (MACD < Signal)"
    text += f"📊 **MACD:** `{last_row['MACD']:.4f}` | **Signal:** `{last_row['MACD_Signal']:.4f}` — {macd_sig}\n"
    
    bb_status = "Верхняя граница" if current_price > last_row['BB_Upper'] else "Нижняя граница" if current_price < last_row['BB_Lower'] else "В пределах канала"
    text += f"📈 **Bollinger Bands:** {bb_status}\n"
    text += f"   Верх: `{last_row['BB_Upper']:.4f}` | Средняя: `{last_row['BB_Middle']:.4f}` | Низ: `{last_row['BB_Lower']:.4f}`\n"
    
    adx_val = last_row['ADX']
    trend_str = "Сильный тренд" if adx_val > MIN_ADX_FOR_TREND else "Слабый тренд/консолидация"
    text += f"📊 **ADX (14):** `{adx_val:.2f}` — {trend_str}\n"
    di_diff = last_row['Plus_DI'] - last_row['Minus_DI']
    di_sig = "Преобладает бычий тренд (+DI > -DI)" if di_diff > 0 else "Преобладает медвежий тренд (-DI > +DI)"
    text += f"   +DI: `{last_row['Plus_DI']:.2f}` | -DI: `{last_row['Minus_DI']:.2f}` — {di_sig}\n\n"
    
    text += "⚡ **РЕКОМЕНДАЦИИ:**\n"
    recs = []
    if rsi_val > 70: recs.append("RSI показывает перекупленность - возможна коррекция")
    elif rsi_val < 30: recs.append("RSI показывает перепроданность - возможен отскок")
    
    if macd_diff > 0 and macd_diff > 0.5 * last_row['ATR']: recs.append("Сильный бычий сигнал MACD")
    elif macd_diff < 0 and abs(macd_diff) > 0.5 * last_row['ATR']: recs.append("Сильный медвежий сигнал MACD")
    
    if current_price > last_row['BB_Upper']: recs.append("Цена выше верхней границы Bollinger Bands - возможен откат")
    elif current_price < last_row['BB_Lower']: recs.append("Цена ниже нижней границы Bollinger Bands - возможен отскок")
    
    if adx_val > MIN_ADX_FOR_TREND:
        if last_row['Plus_DI'] > last_row['Minus_DI']: recs.append("Сильный бычий тренд (ADX > 25 и +DI > -DI)")
        else: recs.append("Сильный медвежий тренд (ADX > 25 и -DI > +DI)")
    else:
        recs.append("Слабый тренд или консолидация (ADX < 25)")
        
    for i, rec in enumerate(recs, 1):
        text += f"{i}. {rec}\n"
        
    return text

def build_entry_points_report(df: pd.DataFrame) -> str:
    current_price = df['close'].iloc[-1]
    df['min'] = df['low'].rolling(10, center=True).min()
    df['max'] = df['high'].rolling(10, center=True).max()
    
    supports = df[df['low'] == df['min']]['low'].unique()
    resistances = df[df['high'] == df['max']]['high'].unique()
    
    nearest_supports = [s for s in supports if s < current_price][-3:]
    nearest_resistances = [r for r in resistances if r > current_price][:3]
    
    text = f"🔍 **Точки входа для {current_price:.4f}**\n\n"
    
    if len(nearest_supports) >= 1:
        main_support = nearest_supports[-1]
        strength = 'Сильная' if len(nearest_supports) >= 2 else 'Умеренная'
        advice = 'Лучшая точка для лимитного ордера' if len(nearest_supports) >= 2 else 'Можно ждать подтверждения'
        text += "🟢 **ЛУЧШИЕ ТОЧКИ ДЛЯ ПОКУПКИ:**\n"
        text += f"• **Цена:** `{main_support:.4f}` | **Тип:** Основная поддержка\n"
        text += f"  **Сила:** {strength} | **Совет:** {advice}\n\n"
        
    if len(nearest_resistances) >= 1:
        main_resistance = nearest_resistances[0]
        strength = 'Сильное' if len(nearest_resistances) >= 2 else 'Умеренное'
        advice = 'Лучшая точка для лимитного ордера' if len(nearest_resistances) >= 2 else 'Можно ждать подтверждения'
        text += "🔴 **ЛУЧШИЕ ТОЧКИ ДЛЯ ПРОДАЖИ:**\n"
        text += f"• **Цена:** `{main_resistance:.4f}` | **Тип:** Основное сопротивление\n"
        text += f"  **Сила:** {strength} | **Совет:** {advice}\n\n"
        
    if not nearest_supports and not nearest_resistances:
        text += "⚠️ Нет четких точек входа. Рынок в консолидации.\n"
        
    return text

def build_limit_orders_report(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    current_price = df['close'].iloc[-1]
    direction = determine_market_direction(df)
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
    step = atr * 0.5 if atr > 0 else current_price * 0.01
    
    text = f"📊 **Лимитные ордера для {symbol} ({timeframe})**\n"
    text += f"📈 **Текущая цена:** `{current_price:.4f}`\n"
    text += f"📌 **ATR:** `{atr:.4f}` (шаг ордеров: `{step:.4f}`)\n\n"
    
    if direction == "BULLISH":
        for i in range(1, 4):
            price = current_price - (step * i)
            vol = f"{100//i}%"
            text += f"🔹 **BUY LIMIT по `{price:.4f}`**\n"
            text += f"   **Объем:** {vol} | Уровень входа #{i} (ATR-based)\n\n"
    elif direction == "BEARISH":
        for i in range(1, 4):
            price = current_price + (step * i)
            vol = f"{100//i}%"
            text += f"🔹 **SELL LIMIT по `{price:.4f}`**\n"
            text += f"   **Объем:** {vol} | Уровень входа #{i} (ATR-based)\n\n"
    else:
        text += "⚠️ Нет активных сигналов для размещения ордеров.\n\n"
        
    text += "💡 **СОВЕТ:** Размещайте ордера с шагом 0.5-1 ATR\n"
    text += "Используйте ступенчатые ордера для усреднения входа"
    
    return text

def generate_chart(df: pd.DataFrame, symbol: str) -> io.BytesIO:
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8), dpi=100)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['open_time'], df['close'], label='Цена', color='cyan')
    if 'SMA_20' in df.columns: ax1.plot(df['open_time'], df['SMA_20'], label='SMA 20', color='orange')
    if 'SMA_50' in df.columns: ax1.plot(df['open_time'], df['SMA_50'], label='SMA 50', color='purple')
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        ax1.plot(df['open_time'], df['BB_Upper'], color='red', linestyle='--', alpha=0.5)
        ax1.plot(df['open_time'], df['BB_Lower'], color='green', linestyle='--', alpha=0.5)
        ax1.fill_between(df['open_time'], df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_title(f"{symbol} Price Analysis")
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if 'MACD' in df.columns:
        ax2.plot(df['open_time'], df['MACD'], color='blue')
        ax2.plot(df['open_time'], df['MACD_Signal'], color='red')
        ax2.bar(df['open_time'], df['MACD_Hist'], color='gray', alpha=0.5)
        ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if 'RSI' in df.columns:
        ax3.plot(df['open_time'], df['RSI'], color='purple')
        ax3.axhline(30, color='green', linestyle='--')
        ax3.axhline(70, color='red', linestyle='--')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if 'ADX' in df.columns:
        ax4.plot(df['open_time'], df['ADX'], color='white', label='ADX')
        ax4.plot(df['open_time'], df['Plus_DI'], color='green', label='+DI')
        ax4.plot(df['open_time'], df['Minus_DI'], color='red', label='-DI')
        ax4.axhline(MIN_ADX_FOR_TREND, color='gray', linestyle='--')
        ax4.grid(True, alpha=0.2)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==================== TELEGRAM ХЕНДЛЕРЫ ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 **Добро пожаловать в бот SignalX BYB!**\n\n"
        "📖 **Инструкция по использованию:**\n"
        "1. Зарегистрируйтесь на бирже Bybit по нашей реферальной ссылке.\n"
        "2. Нажмите кнопку «Подтвердить регистрацию».\n"
        "3. Введите ваш Bybit UID для проверки.\n"
        "4. После подтверждения администратором вам откроется полный функционал!\n\n"
        "📜 **Пользовательское соглашение и Инструкция SignalX:**\n\n"
        "1. **Источник аналитики:** Бот получает котировки и маркет-данные в реальном времени напрямую с платформы **Bybit**.\n\n"
        "2. **Точность сигналов:** Все алгоритмы, расчёты уровней и индикаторы оптимизированы под ликвидность Bybit. На других биржах цены могут незначительно отличаться из-за спредов.\n\n"
        "3. **Рекомендация:** Для наилучшей точности и совпадения точек входа настоятельно рекомендуем торговать на бирже **Bybit**.\n\n"
        "⚠️ *SignalX предоставляет исключительно аналитическую информацию и не является индивидуальной финансовой рекомендацией.*"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔗 Регистрация на Bybit", url=REFERRAL_LINK)],
        [InlineKeyboardButton("✅ Я зарегистрировался (Проверить)", callback_data="check_reg")]
    ]
    await update.message.reply_text(welcome_text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "check_reg":
        await query.message.reply_text("Пожалуйста, отправьте ваш **Bybit UID** (числовой ID профиля) для проверки:")
        context.user_data['awaiting_uid'] = True
        
    elif query.data == "back_to_coins":
        await show_main_menu_from_query(query)
        
    elif query.data.startswith("select_crypto_"):
        symbol = query.data.split("_")[2]
        context.user_data['selected_crypto'] = symbol
        keyboard = [
            [InlineKeyboardButton(tf, callback_data=f"analyze_{symbol}_{tf}") for tf in ["15m", "30m", "1h", "4h"]],
            [InlineKeyboardButton(tf, callback_data=f"analyze_{symbol}_{tf}") for tf in ["1m", "5m", "1d"]],
            [InlineKeyboardButton("🔙 Назад к выбору монеты", callback_data="back_to_coins")]
        ]
        await query.message.reply_text(f"Выбрана пара **{symbol}/USDT**. Выберите таймфрейм:", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif query.data.startswith("analyze_"):
        _, symbol, tf = query.data.split("_")
        await query.message.reply_text(f"⏳ Выполняю полный анализ {symbol} на TF {tf}...")
        
        df = get_historical_data(symbol, TIMEFRAMES[tf])
        if df.empty:
            await query.message.reply_text("❌ Ошибка при получении данных от биржи.")
            return
            
        df = calculate_indicators(df)
        
        # 1. График
        chart_buf = generate_chart(df, symbol)
        
        # 2. Разделы отчета
        analysis_report = build_analysis_report(df, symbol, tf)
        entry_report = build_entry_points_report(df)
        orders_report = build_limit_orders_report(df, symbol, tf)
        
        # Отправка отчетов
        await query.message.reply_photo(photo=chart_buf, caption=f"📈 **График теханализа {symbol} ({tf})**", parse_mode="Markdown")
        await query.message.reply_text(analysis_report, parse_mode="Markdown")
        await query.message.reply_text(entry_report, parse_mode="Markdown")
        
        # Кнопка для перезапуска
        back_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Выбрать другую монету", callback_data="back_to_coins")]
        ])
        await query.message.reply_text(orders_report, parse_mode="Markdown", reply_markup=back_keyboard)

    # =============== МОДЕРАЦИЯ АДМИНИСТРАТОРОМ ===============
    elif query.data.startswith("approve_"):
        target_user_id = int(query.data.split("_")[1])
        REGISTERED_USERS.add(target_user_id)
        
        # Изменяем сообщение администратору
        await query.edit_message_text(f"{query.message.text}\n\n✅ **ОДОБРЕНО**")
        
        # Уведомляем пользователя
        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text="🎉 **Ваша регистрация успешно подтверждена!**\nТеперь вам доступен полный функционал анализа SignalX BYB."
            )
            # Отправляем меню выбора монет
            keyboard = []
            keys = list(CRYPTO_LIST.keys())
            for i in range(0, len(keys), 3):
                row = [InlineKeyboardButton(coin, callback_data=f"select_crypto_{coin}") for coin in keys[i:i+3]]
                keyboard.append(row)
            await context.bot.send_message(
                chat_id=target_user_id,
                text="📊 **Выберите криптовалюту для анализа:**",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            print(f"Ошибка при отправке сообщения пользователю: {e}")

    elif query.data.startswith("reject_"):
        target_user_id = int(query.data.split("_")[1])
        
        # Изменяем сообщение администратору
        await query.edit_message_text(f"{query.message.text}\n\n❌ **ОТКЛОНЕНО**")
        
        # Уведомляем пользователя
        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text="❌ **Регистрация не подтверждена.**\nВаш UID не найден в списке рефералов. Убедитесь, что вы зарегистрировались по нашей ссылке и повторите попытку."
            )
        except Exception as e:
            print(f"Ошибка при отправке сообщения пользователю: {e}")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name
    username = f"@{update.effective_user.username}" if update.effective_user.username else "нет username"
    
    if context.user_data.get('awaiting_uid'):
        uid = update.message.text.strip()
        if uid.isdigit() and len(uid) >= 5:
            context.user_data['awaiting_uid'] = False
            await update.message.reply_text("⏳ **Заявка отправлена на проверку.**\nАдминистратор проверит ваш UID и откроет доступ в ближайшее время.")
            
            # Отправляем заявку администратору
            admin_text = (
                f"📥 **Новая заявка на доступ SignalX BYB!**\n\n"
                f"👤 Пользователь: {user_name} ({username})\n"
                f"🆔 Telegram ID: `{user_id}`\n"
                f"🔑 **Bybit UID:** `{uid}`"
            )
            
            admin_keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{user_id}"),
                    InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{user_id}")
                ]
            ])
            
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=admin_text,
                    parse_mode="Markdown",
                    reply_markup=admin_keyboard
                )
            except Exception as e:
                print(f"Ошибка при отправке заявки админу: {e}")
                await update.message.reply_text("⚠️ Не удалось связаться с администратором. Убедитесь, что ADMIN_ID указан верно.")
        else:
            await update.message.reply_text("❌ Некорректный UID. Пожалуйста, введите корректный Bybit UID (только цифры).")
    else:
        if user_id not in REGISTERED_USERS:
            await start(update, context)
        else:
            await show_main_menu(update)

async def show_main_menu(update: Update):
    keyboard = []
    keys = list(CRYPTO_LIST.keys())
    for i in range(0, len(keys), 3):
        row = [InlineKeyboardButton(coin, callback_data=f"select_crypto_{coin}") for coin in keys[i:i+3]]
        keyboard.append(row)
        
    await update.message.reply_text("📊 **Выберите криптовалюту для анализа:**", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_main_menu_from_query(query):
    keyboard = []
    keys = list(CRYPTO_LIST.keys())
    for i in range(0, len(keys), 3):
        row = [InlineKeyboardButton(coin, callback_data=f"select_crypto_{coin}") for coin in keys[i:i+3]]
        keyboard.append(row)
        
    await query.message.reply_text("📊 **Выберите криптовалюту для анализа:**", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    print("🤖 Бот SignalX BYB запущен!")
    app.run_polling()