import os
import sys
import argparse

# ===== ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ =====
DEFAULT_DIRECTORY = r"D:\git\yolo_train\temp"   # полный путь к каталогу
DEFAULT_PREFIX = "file"                            # префикс для новых имён
DEFAULT_START = 1                                   # начальный номер
# =================================


def rename_files(directory: str, prefix: str, start_number: int = 1):
    """
    Переименовывает файлы в каталоге в формате "префикс_номер.расширение"
    """
    if not os.path.isdir(directory):
        print(f"Ошибка: каталог '{directory}' не существует или не является каталогом.")
        sys.exit(1)

    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    files.sort()

    if not files:
        print("В каталоге нет файлов для переименования.")
        return

    print(f"Каталог: {directory}")
    print(f"Префикс: {prefix}")
    print(f"Найдено файлов: {len(files)}")
    print("-" * 60)

    number = start_number
    for filename in files:
        old_path = os.path.join(directory, filename)
        _, extension = os.path.splitext(filename)

        new_name = f"{prefix}_{number}{extension}"
        new_path = os.path.join(directory, new_name)

        if os.path.exists(new_path) and os.path.abspath(old_path) != os.path.abspath(new_path):
            print(f"[ПРОПУЩЕН] '{filename}' -> '{new_name}' (файл уже существует)")
            continue

        if filename == new_name:
            print(f"[БЕЗ ИЗМЕНЕНИЙ] '{filename}'")
            number += 1
            continue

        try:
            os.rename(old_path, new_path)
            print(f"[OK] '{filename}' -> '{new_name}'")
            number += 1
        except OSError as e:
            print(f"[ОШИБКА] Не удалось переименовать '{filename}': {e}")

    print("-" * 60)
    print("Готово.")


def main():
    parser = argparse.ArgumentParser(
        description="Переименование файлов в каталоге в формате 'префикс_номер.расширение'"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DIRECTORY,
        help=f"Полный путь к каталогу (по умолчанию: {DEFAULT_DIRECTORY})"
    )
    parser.add_argument(
        "-p", "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Префикс для новых имён (по умолчанию: '{DEFAULT_PREFIX}')"
    )
    parser.add_argument(
        "-s", "--start",
        type=int,
        default=DEFAULT_START,
        help=f"Начальный номер (по умолчанию: {DEFAULT_START})"
    )

    args = parser.parse_args()
    rename_files(args.directory, args.prefix, args.start)


if __name__ == "__main__":
    main()