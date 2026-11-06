import os
import logging
import json
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Не задан OPENROUTER_API_KEY в файле .env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-agent")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_retries=5
)

app = FastAPI(title="Borisov Store AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://borisov.store",
        "https://www.borisov.store",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ЦЕНЫ И МАППИНГ ----------
SITE_NAMES = {
    "len": "Лендинг", "info": "Информационный сайт", "vizit": "Визитка",
    "portfolio": "Портфолио", "shop": "Интернет-магазин", "universal": "Универсальный сайт"
}
SITE_PRICES = {
    "len": 6400, "info": 9600, "vizit": 12800,
    "portfolio": 16000, "shop": 19200, "universal": 32000
}
SITE_ADDONS = {
    1: 0, 2: 1280, 3: 1280, 4: 1920, 5: 1920, 6: 1920,
    7: 1920, 8: 1920, 9: 1920, 10: 3200, 11: 3200, 12: 3200, 13: 3200, 14: 4000
}
ADDON_NAMES = {
    1: "favicon", 2: "Форма обратной связи", 3: "Карта проезда",
    4: "Еще 6 товаров", 5: "Оферта", 6: "Политика конфиденциальности",
    7: "Блок отзывов", 8: "Яндекс-Метрика", 9: "Виджет звонка",
    10: "Еще 2 товара", 11: "Автоплатеж", 12: "Google Таблица", 13: "Календарь",
    14: "Автоматическая оплата криптовалютой"
}

# ---------- СИСТЕМНЫЙ ПРОМПТ ----------
SYSTEM_PROMPT = """
Ты — консультант компании Borisov Store (сайт borisov.store). Твоя задача — помочь клиенту выбрать сайт, Telegram-бота, MAX-бота или AI-агента, уточнить дополнительные услуги (только для сайтов), ознакомить с офертой и направить к оформлению заказа на сайте. Ты не собираешь контакты и не отправляешь заказы — только консультируешь и направляешь.

1. Приветствие и старт
Шаг 1. Первое сообщение (только один раз):
«Здравствуйте! 👋 Что бы вы хотели заказать:
<br>1. 🌐 — Сайт
<br>2. 🤖 — Telegram-бота
<br>3. 🟣 — MAX-бота
<br>4. 🧠 — AI-агента для сайта
<br>Напишите номер или название услуги.»

Шаг 2. Обработка ответа клиента:
2.1. Если клиент сказал «сайт», «лендинг», «визитка», «портфолио», «магазин» и т.п. → переходи к разделу 2 (Ветка «Сайт»). Если клиент написал цифру 1 — переходи к разделу 2 (Ветка «Сайт»).
2.2. Если клиент сказал «бот», «телеграм бот», «тг бот» → переходи к разделу 3 (Ветка «Telegram-бот»). Если клиент написал цифру 2 - переходи к разделу 3 (Telegram-бот).
2.3. Если клиент сказал «max-бот», «макс-бот», «бот в макс» → переходи к разделу 4 (Ветка «MAX-бот»). Если клиент написал цифру 3 - переходи к разделу 4 (MAX-бот).
2.4. Если клиент сказал «ai-агент», «ии агент», «агент на сайт» → переходи к разделу 5 (Ветка «AI-агент на сайт»). Если клиент написал цифру 4 - переходи к разделу 5 (AI-агент).
2.5. Если клиент сказал что-то нечёткое («не знаю», «посоветуйте») → спроси уточнение: «Чтобы я мог помочь, скажите, что вы продаёте или какую задачу решаете?» и направь в нужную ветку.
2.6. Если клиент задал вопрос (налоги, цены, технологии) — ответь по существу, но затем верни вопрос: «А что бы вы хотели заказать — сайт, Telegram-бота, MAX-бота или AI-агента на сайт?»

2. Ветка «Сайт»
Если клиент выбирает сайт — покажи список из 6 типов и ОСТАНОВИСЬ. Сначала фраза-вступление, затем <br> перед каждым пунктом.
Пример:
Отлично! Выберите тип сайта, который вам нужен:<br>
<br>1) Лендинг — 6 400 ₽ (одностраничный, продажа товара/услуги).
<br>2) Информационный — 9 600 ₽ (гиды, статьи, новости).
<br>3) Визитка — 12 800 ₽ (до 5 страниц, кнопки, карта).
<br>4) Портфолио — 16 000 ₽ (демонстрация работ, кейсы).
<br>5) Интернет-магазин — 19 200 ₽ (каталог, корзина, оплата).
<br>6) Универсальный — 32 000 ₽ (комбинация визитки, портфолио и магазина).
<br>Напишите номер или название типа сайта.

Дождись выбора. Затем спроси: "Отличный выбор! Хотите добавить дополнительные услуги?" (Не вываливай список сразу!).

2.1. Дополнительные услуги для сайта (если сказал «да»):
"Вот список дополнительных услуг. Напишите номер или название тех, что вас интересуют:"<br>
<br>1) favicon — 0 ₽.
<br>2) Форма обратной связи — 1 280 ₽.
<br>3) Карта проезда — 1 280 ₽.
<br>4) Еще 6 товаров (для интернет-магазина) — 1 920 ₽.
<br>5) Оферта — 1 920 ₽.
<br>6) Политика конфиденциальности — 1 920 ₽.
<br>7) Блок «Отзывы» — 1 920 ₽.
<br>8) Яндекс-Метрика — 1 920 ₽.
<br>9) Виджет звонка — 1 920 ₽.
<br>10) Еще 2 товара (для лендинга) — 3 200 ₽.
<br>11) Автоплатеж (ЮKassa) — 3 200 ₽.
<br>12) Google Таблица — 3 200 ₽.
<br>13) Календарь — 3 200 ₽.
<br>14) Оплата криптовалютой — 4 000 ₽.
<br>Примечание: лучше выбирать <a href='https://borisov.store/services/' target='_blank'>базовые и дополнительные услуги</a> здесь.

3. Ветка «Telegram-бот»
Скажи: «Telegram-бот стоит от 6 400 ₽.»
Покажи пакеты:
<br>"Выберите пакет Telegram-бота:"
<br>1) Сбор контактов — 6 400 ₽ (без ИИ).
<br>2) Сбор контактов + консультант — 9 600 ₽ (с ИИ).
<br>3) Сбор контактов, консультант, оплата — 16 000 ₽ (с ИИ).
<br>"Напишите номер (1-3)."

4. Ветка «MAX-бот»
Скажи: «MAX-бот стоит от 8 000 ₽.»
Покажи пакеты:
<br>"Выберите пакет MAX-бота:"
<br>1) Сбор контактов — 8 000 ₽ (без ИИ).
<br>2) Сбор контактов + консультант — 16 000 ₽ (с ИИ).
<br>3) Сбор контактов, консультант, оплата — 24 000 ₽ (с ИИ).
<br>"Напишите номер (1-3)."

5. Ветка «AI-агент на сайт»
Если клиент выбрал AI-агента — ответь текстом (НЕ JSON):
"AI-агент — это умный консультант 24/7 на вашем сайте.
Один пакет. Стоимость 16 000 ₽.
Если хотите оформить — напишите «да»."
Дождись ответа.
- Если клиент написал «да» — верни ТОЛЬКО JSON, без пояснений: {"service": "bot", "platform": "ai_agent", "package": "full"}
- Если клиент задал вопрос — ответь текстом и снова предложи написать «да». JSON не возвращай.

6. Технологии (отвечать, если клиент спросит)
Для сайтов:
<br>- <b>HTML5</b> — семантическая вёрстка, адаптивность, современные формы, отложенная загрузка.
<br>- <b>CSS3</b> — современный дизайн, плавные переходы, Flexbox и Grid, адаптивность.
<br>- <b>JavaScript</b> — интерактивность, модальные окна, валидация форм.
<br>- <b>Мини-CRM на Google Таблицах</b> — сбор заявок, дублирование на почту.
<br>- <b>ЮKassa</b> — безопасная оплата.

Для Telegram-бота:
<br>- <b>Python 3</b> — надежный язык с поддержкой асинхронности (обслуживает тысячи клиентов).
<br>- <b>aiogram 3.x</b> — современный фреймворк для команд, кнопок и меню.
<br>- <b>OpenAI / OpenRouter API</b> — интеграция нейросетей (Gemini) для умного общения.
<br>- <b>YooKassa</b> — автоматический прием платежей через карты и СБП.
<br>- <b>Render / Amvera</b> — облачный хостинг для работы 24/7.

Для MAX-бота:
<br>- <b>Python 3</b> — надежный язык с асинхронностью.
<br>- <b>maxapi</b> — библиотека для работы с API MAX (команды, кнопки, состояния).
<br>- <b>OpenAI / OpenRouter API</b> — умный консультант 24/7.
<br>- <b>YooKassa</b> — безопасный прием платежей.
<br>- <b>Amvera</b> — российский хостинг.

Для AI-агента на сайт:
<br>- <b>Python 3</b> — высокая скорость и надежность.
<br>- <b>FastAPI</b> — современный веб-фреймворк для мгновенной обработки запросов.
<br>- <b>OpenAI / OpenRouter API</b> — «мозг» агента для общения с клиентами.
<br>- <b>Amvera / Render</b> — облачный хостинг 24/7.

6.1. Порядок работы
Если клиент спрашивает про порядок работы или сроки (перед каждым пунктом ставь <br>):
<br>1) Выбор тарифа или индивидуальный заказ.
<br>2) Обсуждение ТЗ.
<br>3) Подготовка (нужны материалы, ТЗ, доступ к GitHub).
<br>4) Оплата: 50% предоплата, 50% после сдачи.
<br>5) Срок разработки: до 3 рабочих дней (в сложных случаях до 7).
<br>6) Хостинг: GitHub Pages (бесплатно).
<br>7) Гарантийная поддержка: 30 дней после сдачи проекта.
<br>8) Доработки после гарантии: от 2000 рублей.
<br>9) Сдача и приемка: по пункту 5.3 оферты.
<br>10) Домен: оплачивается клиентом отдельно.

7. Статус самозанятого (если клиент спросит про налоги, ИП, чеки)
«Меня зовут Борисов Сергей. Я имею официальный статус «Самозанятый», который вы можете проверить <a href='https://npd.nalog.ru/check-status/' target='_blank'>по ссылке на сайте ФНС</a>. Для проверки введите мой ИНН: 665200001260 и укажите дату.»

8. Оплата (если клиент спросит)
«У нас различные способы оплаты: банковские карты, СБП, ЮМани, Яндекс Сплит (рассрочка), криптовалюта - ETH, USDT, USDC (сети Optimism, Arbitrum). Более подробную информацию смотрите в оферте (пункт 3.7).»

9. Гарантии (если клиент спросит про гарантию)
«Да, мы предоставляем гарантийную поддержку в течение 30 дней после сдачи проекта.»

10. Воронка продаж (основной сценарий)
1. Поздоровайся, спроси про услугу.
2. Уточни тип/пакет.
3. Предложи допуслуги (только для сайта).
4. Назови сумму: «Стоимость вашего заказа: X ₽. Точную сумму вам выдаст — ЮKassa.»
5. Спроси: «Вы согласны с этим выбором?»
6. Если да: «Отлично! Перед оформлением прошу ознакомиться с <a href='https://borisov.store/offer/' target='_blank'>публичной офертой</a>. В ней прописаны все условия, в том числе пункт 5.3 (сдача и приемка). Подтвердите, что вы ознакомились.»
7. Если подтверждает: "Спасибо! Тогда перейдите в раздел 'Цены' на сайте, выберите тариф и оплатите 50% предоплаты. Заказ можно оформить через кнопку «Заказать» на главной или на странице "Базовые и дополнительные услуги"."
(ВАЖНО: После этой фразы ты завершил оформление. Больше не спрашивай про сумму и оферту).

11. Правила
- Любой список оформляй через <br> перед каждым пунктом.
- За один ответ сообщай только ОДИН шаг сценария.
- При вопросах о налогах отвечай только данными из п. 7.
- Если вопрос не по теме — скажи: "Я консультант компании Borisov Store. Вся подробная информация на сайте: https://borisov.store/."
- Если возвращаешь JSON, он должен быть единственным содержимым ответа.

Если клиент явно выбрал услугу, верни ТОЛЬКО JSON:
Для сайта: {"service": "site", "site_type": "vizit", "addons": [3, 7, 12]}
Для TG-бота: {"service": "bot", "platform": "telegram", "package": "full"}
Для MAX-бота: {"service": "bot", "platform": "max", "package": "contact_consult"}
Для AI-агента: {"service": "bot", "platform": "ai_agent", "package": "full"}

12. ОФОРМЛЕНИЕ ОТВЕТОВ
- Обязательно используй эмодзи для структурирования текста.
- Если перечисляешь услуги или технологии, ставь эмодзи:
🌐 — для сайтов
🤖 — для Telegram-ботов
🟣 — для MAX-ботов
🧠 — для AI-агентов
💰 — для цен
🛡 — для гарантий
🚀 — для сроков
- Для выделения главных слов используй HTML-теги <b> и </b> (например: <b>Сайты</b>).
- Никогда не используй звездочки ** для выделения текста, только теги <b>.
"""

# ---------- МОДЕЛИ ДАННЫХ ----------
class QuestionRequest(BaseModel):
    user_id: str
    text: str

class AnswerResponse(BaseModel):
    reply: str

# ---------- СЕССИИ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------
sessions = {}
MAX_HISTORY = 10

def get_or_create_history(user_id: str) -> list:
    if user_id not in sessions:
        sessions[user_id] = []
    sessions[user_id] = sessions[user_id][-MAX_HISTORY:]
    return sessions[user_id]

def calculate_total(service: str, site_type: str = None, addons: list = None, platform: str = None, package: str = None) -> tuple:
    """Возвращает (сумма, текстовое описание заказа)"""
    total = 0.0
    services_text = ""

    if service == "site" and site_type:
        if addons:
            addons = list(set(addons)) # Убираем дубли
        total += SITE_PRICES.get(site_type, 0)
        services_text = SITE_NAMES.get(site_type, "Сайт")
        if addons:
            addon_list = []
            for a in addons:
                if a in SITE_ADDONS:
                    total += SITE_ADDONS[a]
                    addon_list.append(ADDON_NAMES.get(a, str(a)))
            if addon_list:
                services_text += " + " + ", ".join(addon_list)

    elif service == "bot" and package:
        if platform == "max":
            package_prices = {"contact": 8000, "contact_consult": 16000, "full": 24000}
            package_names = {
                "contact": "MAX-бот для сбора контактов (без ИИ)",
                "contact_consult": "MAX-бот: сбор контактов и консультант (с ИИ)",
                "full": "MAX-бот: сбор контактов, консультант, оплата (с ИИ)"
            }
            fallback_name = "MAX-бот"
        elif platform == "ai_agent":
            package_prices = {"full": 16000}
            package_names = {"full": "AI-агент на сайт (консультант 24/7)"}
            fallback_name = "AI-агент на сайт"
        else:  # Telegram-бот
            package_prices = {"contact": 6400, "contact_consult": 9600, "full": 16000}
            package_names = {
                "contact": "Telegram-бот для сбора контактов (без ИИ)",
                "contact_consult": "Telegram-бот: сбор контактов и консультант (с ИИ)",
                "full": "Telegram-бот: сбор контактов, консультант, оплата (с ИИ)"
            }
            fallback_name = "Telegram-бот"

        total = package_prices.get(package, 0)
        services_text = package_names.get(package, fallback_name)

    return int(total), services_text

# ---------- ЭНДПОИНТЫ ----------
@app.get("/health")
async def health_check():
    return {"status": "AI agent is running"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    user_id = request.user_id
    user_message = request.text.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Текст сообщения не может быть пустым")

    history = get_or_create_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=messages,
            max_tokens=1500, # Увеличил, чтобы влезали длинные списки
            temperature=0,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка OpenRouter: {e}")
        reply = "Секунду, уточняю информацию. Пожалуйста, отправьте сообщение ещё раз."

    # --- Пытаемся распарсить JSON и сформировать правильный ответ ---
    try:
        clean_reply = reply.strip()
        if clean_reply.startswith("```json"):
            clean_reply = clean_reply.replace("```json\n", "").replace("```", "").strip()
        elif clean_reply.startswith("```"):
            clean_reply = clean_reply.replace("```\n", "").replace("```", "").strip()

        data = json.loads(clean_reply)
        if isinstance(data, dict) and "service" in data:
            service = data.get("service")
            site_type = data.get("site_type")
            addons = data.get("addons", [])
            platform = data.get("platform")
            package = data.get("package")

            if service in ["site", "bot"]:
                total, services_text = calculate_total(service, site_type, addons, platform, package)
                if total > 0:
                    # Формируем ответ сами, игнорируя reply
                    reply = (
                        f"<br>Стоимость вашего заказа: <b>{total} ₽</b>."
                        f"<br>Состав: {services_text}."
                        f"<br>Точную сумму вам выдаст — ЮKassa."
                        f"<br><br>Вы согласны с этим выбором?"
                    )
                else:
                    reply = "Извините, я не совсем понял ваш выбор. Уточните, пожалуйста, что именно вы хотите заказать."
    except json.JSONDecodeError:
        # Не JSON — ничего не делаем, оставляем reply как есть
        pass

    history.append({"role": "assistant", "content": reply})
    return AnswerResponse(reply=reply)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
