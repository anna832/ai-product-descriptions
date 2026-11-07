import json
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.environ["OLLAMA_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
TIMEOUT = int(os.environ["TIMEOUT"])
TEMPERATURE = float(os.environ["TEMPERATURE"])
NUM_CTX = int(os.environ["NUM_CTX"])

def load_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception("Файл prompt.txt не найден")

SYSTEM_PROMPT = load_prompt()

def validate_product(product, index):
    errors = []
    warnings = []

    # Обязательное поле
    if not product.get("name"):
        errors.append(f"Товар #{index}: отсутствует название")

    # Желательные поля
    if not product.get("manufacturer"):
        warnings.append(f"Товар #{index}: отсутствует производитель")

    if not product.get("specs"):
        warnings.append(f"Товар #{index}: отсутствуют характеристики")

    if not product.get("description"):
        warnings.append(f"Товар #{index}: отсутствует описание")

    return errors, warnings


def generate_card(product):

    prompt = f"""{SYSTEM_PROMPT}

Данные товара:
{json.dumps(product, ensure_ascii=False, indent=2)}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()

        result = response.json()
        raw_text = result.get("response", "")

        # Модель может добавить лишние слова, поэтому ищем json в тексте
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1

            if start != -1 and end != 0:
                json_str = raw_text[start:end]
                card = json.loads(json_str)
                return {"success": True, "card": card, "raw": raw_text}
            else:
                return {"success": False, "error": "Модель не вернула JSON", "raw": raw_text}

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Ошибка парсинга JSON: {e}", "raw": raw_text}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Превышено время ожидания"}

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Ошибка запроса к Ollama: {e}"}


def main():
    print("Запуск генератора карточек товаров...")

    try:
        with open("input.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("!!! Файл input.json не найден")
        return
    except json.JSONDecodeError as e:
        print(f"!!! Ошибка в формате JSON: {e}")
        return

    if isinstance(data, dict):
        products = [data]
    elif isinstance(data, list):
        products = data[:2]
    else:
        print("!!! Неверный формат данных")
        return

    print(f"Найдено товаров: {len(products)}")

    results = []

    for i, product in enumerate(products, 1):
        print(f"\nОбрабатываю товар #{i}: {product.get('name', 'Без названия')}")

        errors, warnings = validate_product(product, i)

        if errors:
            print(f"!!! Ошибки: {errors}")
            results.append({
                "index": i,
                "name": product.get("name"),
                "status": "error",
                "errors": errors
            })
            continue

        if warnings:
            print(f"! Предупреждения: {warnings}")

        print("Ожидание ответа от модели (может занять до 2 минут)...")
        start_time = time.time()

        result = generate_card(product)

        elapsed = time.time() - start_time
        print(f"Получен ответ за {elapsed:.1f} секунд")

        if result["success"]:
            results.append({
                "index": i,
                "name": product.get("name"),
                "status": "success",
                "warnings": warnings,
                "card": result["card"]
            })
            print(f"Карточка успешно сгенерирована")
        else:
            results.append({
                "index": i,
                "name": product.get("name"),
                "status": "error",
                "errors": [result["error"]],
                "raw_response": result.get("raw", "")
            })
            print(f"Ошибка: {result['error']}")

    output = {
        "total_products": len(products),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Результат сохранён в output.json")
    print(f"Успешно: {output['successful']} из {output['total_products']}")


if __name__ == "__main__":
    main()