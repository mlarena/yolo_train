# SETUP — что установить

Файл описывает установку окружения для обучения YOLO11 / YOLO26. Пакеты **не установлены** —
выполните команды ниже вручную.

---

## 1. Что нужно

| Компонент | Требование |
|---|---|
| Python | 3.10–3.12 (рекомендуется), 3.13 обычно тоже работает |
| RAM | 8 ГБ минимум, 16 ГБ комфортно (`--cache ram` требует больше) |
| Диск | ~15 ГБ (torch + CUDA-библиотеки + датасеты + веса) |
| GPU | NVIDIA + драйвер, поддерживающий нужную версию CUDA (опционально, но сильно быстрее) |
| CPU | Работает, но обучение на тысячах изображений займёт дни |

Проверка GPU (Windows и Linux):

```bash
nvidia-smi
```

Если команда не найдена или видеокарта не NVIDIA — обучение пойдёт на CPU.

---

## 2. Виртуальное окружение

### 2.1 Зачем оно нужно

Пакеты ставятся не в систему, а в отдельный каталог `.venv` внутри проекта. Это даёт:

- изоляцию: `ultralytics`/`torch` не конфликтуют с другими проектами и системным Python;
- чистый перенос: каталог `.venv` не копируется на другую машину, он создаётся заново;
- безопасное удаление: удалили `.venv` — откатили всё, что ставили.

Каталог `.venv` **не** коммитится в git и **не** переносится на другую машину —
там окружение создаётся с нуля (см. 2.3).

### 2.2 Создание в Windows

Из корня проекта:

```powershell
cd <путь к проекту>
py -3.13 -m venv .venv
```

- `py -3.13` — запускает конкретную версию Python; список установленных версий: `py -0p`.
  Можно писать просто `python -m venv .venv` — тогда берётся версия из `PATH`.
- Второй аргумент `.venv` — имя каталога окружения (можно любое, но `.venv` — стандарт,
  его понимают VS Code и большинство инструментов).
- Команда ничего не скачивает, создание занимает несколько секунд.

Проверить, что окружение создалось и в нём правильная версия:

```powershell
.\.venv\Scripts\python.exe --version
```

### 2.3 Создание в Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
cd <путь к проекту>
python3 -m venv .venv
.venv/bin/python --version
```

Пакет `python3-venv` обязателен: без него `python3 -m venv` падает с ошибкой
`ensurepip is not available`.

Если нужна конкретная версия (например, 3.12):

```bash
sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv .venv
```

### 2.4 Активация окружения

Активация нужна, чтобы команды `python` и `pip` указывали внутрь `.venv`.
В начале строки приглашения появится `(.venv)`.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Если PowerShell блокирует запуск скриптов (`running scripts is disabled`):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Windows CMD:

```bat
.venv\Scripts\activate.bat
```

Debian / Linux / macOS:

```bash
source .venv/bin/activate
```

Проверка, что активировано именно окружение:

```bash
python -c "import sys; print(sys.prefix)"
```

Путь должен указывать на `<корень проекта>/.venv`. Если нет — окружение не активировано.

Выйти из окружения: `deactivate`.

> Активация действует только в текущем окне терминала. В новом окне её нужно повторить.
> Альтернатива без активации — вызывать интерпретатор напрямую:
> `.\.venv\Scripts\python.exe train.py` (Windows) или `.venv/bin/python train.py` (Debian).

### 2.5 Пересоздать или удалить окружение

Если окружение сломалось или нужно сменить версию Python:

```powershell
# Windows
deactivate
Remove-Item -Recurse -Force .venv
py -3.13 -m venv .venv
```

```bash
# Debian
deactivate
rm -rf .venv
python3 -m venv .venv
```

После пересоздания зависимости нужно установить заново (разделы 3 или 4).
Каталог `result/` с результатами обучения и каталог с датасетами при этом не затрагиваются.

### 2.6 Выбрать окружение в VS Code

`Ctrl+Shift+P` → `Python: Select Interpreter` → указать
`<корень проекта>\.venv\Scripts\python.exe`. После этого новый терминал VS Code
будет активировать окружение автоматически.

---

## 3. Windows

Окружение `.venv` создаётся в корне проекта (раздел 2.2). Активируйте его по разделу 2.4
и переходите к установке пакетов.

### 3.1 Установить PyTorch

**Вариант A — CPU** (если NVIDIA-карты нет):

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Вариант B — GPU с CUDA 12.8** (актуальную команду смотрите
на https://pytorch.org/get-started/locally/):

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3.2 Установить остальные зависимости

```powershell
python -m pip install -r requirements.txt
```

`ultralytics` подтянет opencv, numpy, pyyaml, matplotlib и т.д. Уже установленный torch
он не перезапишет.

### 3.3 Проверить установку

```powershell
python -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
```

Ожидаемый вывод (для CPU): `cuda False`.

---

## 4. Debian

Окружение создаётся по разделу 2.3, затем активируется:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

**GPU с CUDA 12.8:**

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Только CPU** (не тяните многогигабайтные CUDA-колёса):

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Затем:

```bash
python -m pip install -r requirements.txt
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

Для работы в фоне (обучение идёт часами):

```bash
nohup python train.py --device 0 > train.log 2>&1 &
tail -f train.log
```

Перед запуском проверьте `config.json` в корне проекта — оттуда `train.py` берёт путь
к датасету, а результаты складывает в
`result/<имя датасета>/<имя модели>/runs/<имя прогона>/`.

---

## 5. Предобученные веса (необязательно, но удобно)

`train.py` по умолчанию использует `yolo11n.pt`. Если файла нет, Ultralytics скачает его
сам при первом запуске — но лучше сделать это заранее, одной командой и с прогрессом.
Скрипт `download_models.py` использует только стандартную библиотеку, `ultralytics`
для него не нужен:

```bash
python download_models.py                   # yolo11: n, s, m, l, x (~221 МБ)
python download_models.py --family yolo26   # yolo26: n, s, m, l, x (~231 МБ)
python download_models.py --family both     # обе линейки
python download_models.py --models n s      # только нужные размеры
python download_models.py --models yolo11n yolo26n   # явные имена
python download_models.py --check           # проверить ссылки, ничего не качая
python download_models.py --force           # перекачать
python download_models.py --dir weights     # в отдельный каталог
```

Файлы сохраняются в каталог проекта (или в `--dir`). Скачанные один раз на одной машине,
их можно просто скопировать на другую вместе с проектом — интернет для обучения тогда
не нужен. `*.pt` в `.gitignore`, в git они не попадают.

Проверка целостности (должно быть `True`):

```bash
python -c "import zipfile; print(zipfile.is_zipfile('yolo11n.pt'))"
```

### YOLO26

Для YOLO26 (`yolo26n/s/m/l/x.pt`) нужен `ultralytics>=8.4.0` — это уже указано в
`requirements.txt`. Версия 8.3 не знает такие чекпоинты и падает при загрузке.
Проверить установленную версию:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

Если версия ниже 8.4:

```bash
python -m pip install -U "ultralytics>=8.4.0"
```

Сам `train.py` менять не нужно: `python train.py --model yolo26n.pt`.

---

## 6. Если колёса не ставятся

- `Could not find a version that satisfies the requirement torch` — версия Python слишком новая
  для текущих колёс. Пересоздайте окружение на 3.12 (см. 2.5):
  - Windows: `py -3.12 -m venv .venv` (список версий: `py -0p`)
  - Debian: `sudo apt install -y python3.12-venv` и `python3.12 -m venv .venv`
- `ensurepip is not available` при `python3 -m venv` — не установлен `python3-venv`:
  `sudo apt install -y python3-venv`.
- `No module named venv` — то же самое: `sudo apt install -y python3-venv` (Debian)
  или переустановите Python с компонентом `pip`/`venv` (Windows).
- `Activate.ps1 cannot be loaded because running scripts is disabled` — выполните
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` в этом окне PowerShell.
- `torch.cuda.is_available()` возвращает `False` при наличии GPU:
  1. проверьте `nvidia-smi`;
  2. переустановите torch под CUDA, которую поддерживает драйвер:
     `pip uninstall -y torch torchvision` и повторите установку с нужным `--index-url`;
  3. на Linux проверьте, что стоит не CPU-сборка (`pip show torch`).
- Долгая установка/скачивание — это нормально: CUDA-сборка torch весит несколько ГБ.
  Веса `yolo*.pt` можно скачать заранее: `python download_models.py` (раздел 5).
