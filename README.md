# Garbage Detection — обучение YOLO11

Детекция мусора на изображениях: одна модель, один класс `garbage`.
Датасет Roboflow v8 (CC BY 4.0): 4730 изображений train, 629 valid, 418 test.

---

## 1. Состав проекта

```
yolo_project/
├── config.json                      # НАСТРОЙКА: путь к датасету для train.py
├── datasets/                        # все датасеты (в git не попадают)
│   ├── Garbage Detection.v8i.yolov11/
│   │   ├── data.yaml                # конфиг: пути к сплитам, классы
│   │   ├── train/{images,labels}/
│   │   ├── valid/{images,labels}/
│   │   └── test/{images,labels}/
│   ├── Road-damage.v3i.yolov11/
│   └── Road-damage.v3i.yolo26/
├── result/                          # результаты обучения (в git не попадают)
│   └── <датасет>/<модель>/runs/<прогон>/
├── train.py                         # скрипт обучения (YOLO11 и YOLO26)
├── download_models.py               # предзагрузка весов yolo11/yolo26
├── yolo11n.pt ... yolo11x.pt        # веса YOLO11 (5.4–109 МБ)
├── yolo26n.pt ... yolo26x.pt        # веса YOLO26 (5.3–113 МБ)
├── requirements.txt                 # python-зависимости
├── SETUP.md                         # что установить (Windows / Debian)
├── README.md                        # этот файл
├── .gitignore                       # что не попадает в git
└── .venv/                           # виртуальное окружение
```

Результаты обучения складываются в `result/<имя датасета>/<имя модели>/runs/<имя прогона>/`.
В пути есть и датасет, и модель, поэтому прогоны разных экспериментов не смешиваются:

```
result/
├── Garbage Detection.v8i.yolov11/
│   ├── yolo11n/runs/train/          # python train.py
│   ├── yolo26n/runs/train/          # --model yolo26n.pt
│   └── yolo26s/runs/exp2/           # --model yolo26s.pt --name exp2
└── Road-damage.v3i.yolov11/
    └── yolo26n/runs/train/          # --data datasets/Road-damage.v3i.yolov11/data.yaml
```

Имя модели берётся из `--model`: `yolo26s.pt` → `yolo26s`, `yolo26n-p2.yaml` → `yolo26n-p2`.
При дообучении от готового чекпоинта (`--model result/.../yolo11s/runs/train/weights/best.pt`)
имя модели сохраняется — новый прогон попадёт в `yolo11s/runs/<новый --name>`.

В git не попадают: `.venv/`, `result/`, веса (`*.pt`, `*.onnx`, ...), логи, `__pycache__/`
и весь каталог `datasets/` (десятки МБ изображений) — датасеты хранятся отдельно.
Если датасет всё же нужен в репозитории, уберите его из
`.gitignore` и подключите Git LFS (команда-подсказка есть в комментарии внутри файла).

---

## 2. Установка окружения

См. **SETUP.md**. Кратко для Windows (создание окружения + установка):

```powershell
py -3.13 -m venv .venv                       # создать окружение (один раз)
.\.venv\Scripts\Activate.ps1                 # активировать
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Для Debian:

```bash
sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv                        # создать окружение (один раз)
source .venv/bin/activate                    # активировать
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # или .../whl/cpu
python -m pip install -r requirements.txt
```

Подробно про виртуальное окружение (создание, активация, пересоздание, VS Code,
типичные ошибки) — раздел 2 в **SETUP.md**.

---

## 3. Предобученные веса (скачать заранее)

`train.py` берёт готовый чекпоинт (`--model yolo11n.pt`). Чтобы Ultralytics не качал его
при первом запуске, веса скачиваются заранее скриптом `download_models.py` —
он использует только стандартную библиотеку Python, `ultralytics` для этого не нужен.

```bash
python download_models.py                 # yolo11: n, s, m, l, x
python download_models.py --family yolo26 # yolo26: n, s, m, l, x
python download_models.py --family both   # обе линейки
python download_models.py --models n s    # только нужные размеры
python download_models.py --models yolo11n yolo26n   # явные имена
python download_models.py --check         # только проверить доступность ссылок
python download_models.py --force         # перекачать существующие файлы
python download_models.py --dir weights   # сохранить в ./weights
```

Файлы кладутся в корень проекта, рядом с `train.py`.

### 3.1 YOLO11 (по умолчанию в `train.py`)

| Файл | Размер | Параметры | mAP50-95 (COCO) | Когда брать |
|---|---|---|---|---|
| `yolo11n.pt` | 5.4 МБ | 2.6M | 39.5 | быстрый черновик, слабый GPU/CPU |
| `yolo11s.pt` | 18.4 МБ | 9.4M | 47.0 | баланс скорость/качество |
| `yolo11m.pt` | 38.8 МБ | 20.1M | 51.5 | хороший компромисс для 5 тыс. изображений |
| `yolo11l.pt` | 49.0 МБ | 25.3M | 53.4 | максимум качества при наличии GPU |
| `yolo11x.pt` | 109.3 МБ | 56.9M | 54.7 | только если данных много и есть мощная GPU |

### 3.2 YOLO26 (новее, точнее при том же размере)

| Файл | Размер | Параметры | mAP50-95 (COCO) | e2e (NMS-free) |
|---|---|---|---|---|
| `yolo26n.pt` | 5.3 МБ | 2.4M | 40.9 | 40.1 |
| `yolo26s.pt` | 19.5 МБ | 9.5M | 48.6 | 47.8 |
| `yolo26m.pt` | 42.2 МБ | 20.4M | 53.1 | 52.5 |
| `yolo26l.pt` | 50.7 МБ | 24.8M | 55.0 | 54.4 |
| `yolo26x.pt` | 113.2 МБ | 55.7M | 57.5 | 56.9 |

Размеры и метрики — официальные данные Ultralytics для COCO; на вашем датасете
абсолютные значения будут другими, важен порядок «n < s < m < l < x».
`yolo26n` при почти том же размере, что `yolo11n`, даёт +1.4 mAP и заметно быстрее
на CPU (ONNX), поэтому для нового проекта разумно начинать именно с него.

Скачанные файлы уже лежат в проекте (обе линейки, все десять). Использование:

```bash
python train.py --model yolo11s.pt
python train.py --model yolo26n.pt             # YOLO26, требует ultralytics>=8.4.0
python train.py --model weights/yolo11m.pt     # если скачивали с --dir weights
```

`*.pt` и `weights/` указаны в `.gitignore`, в git они не попадут.

Если веса положить в другое место, Ultralytics ищет их в текущем каталоге и в
`weights_dir` из настроек (`yolo settings`). Самый надёжный вариант — передавать
абсолютный путь в `--model`.

Проверить, что файл не битый (должно быть `True`):

```bash
python -c "import zipfile; print(zipfile.is_zipfile('yolo11n.pt'))"
```

---

## 4. Запуск обучения

### 4.1 Настройка `config.json` — путь к датасету

`train.py` не хранит путь к данным в коде: он читает его из `config.json` в корне проекта.

```json
{
  "data": "datasets/Garbage Detection.v8i.yolov11/data.yaml"
}
```

Правила:

- единственный обязательный ключ — `data`, путь к `data.yaml` датасета;
- относительный путь считается от корня проекта (можно и абсолютный);
- имя каталога результатов берётся автоматически из имени папки датасета
  (`.../datasets/Road-damage.v3i.yolov11/data.yaml` → `result/Road-damage.v3i.yolov11/`),
  поэтому в конфиге ничего дублировать не нужно;
- пустой файл, отсутствующий ключ `data` или битый JSON → скрипт завершится с кодом 2
  и подскажет, что записать;
- разово переопределить датасет без правки конфига: `--data "datasets/.../data.yaml"`.

Сменить датасет — отредактировать одну строку:

```json
{"data": "datasets/Road-damage.v3i.yolov11/data.yaml"}
```

Проверить, что скрипт видит конфиг, можно по первым строкам вывода:

```
[info] config    : E:\yolo_project\config.json
[info] dataset   : Road-damage.v3i.yolov11
[info] model dir : yolo26s
[info] results   : E:\yolo_project\result\Road-damage.v3i.yolov11\yolo26s\runs	rain
```

### 4.2 Проверка, что всё на месте (2–5 минут)

```bash
python train.py --epochs 2 --imgsz 320 --batch 4 --name smoke_test
```

Скрипт печатает сводку по датасету, устройство и стартует обучение. Если 2 эпохи
прошли без ошибок — можно запускать полный прогон.

### 4.3 Полное обучение

```bash
python train.py
```

Значения по умолчанию: `yolo11n.pt`, 100 эпох, `imgsz=640`, `batch=16`,
устройство выбирается автоматически (GPU, если доступен, иначе CPU).
Результаты — в `result/<имя датасета>/<имя модели>/runs/train/`.

### 4.4 Типовые варианты

```bash
# GPU, модель побольше и подольше
python train.py --model yolo26s.pt --epochs 200 --batch 16 --device 0

# CPU / мало памяти
python train.py --batch 4 --workers 2 --device cpu

# две GPU
python train.py --device 0,1 --batch 32

# свой путь к датасету
python train.py --data /data/datasets/garbage/data.yaml

# продолжить прерванное обучение
python train.py --resume
```

### 4.5 Параметры `train.py`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--model` | `yolo11n.pt` | предобученные веса: `yolo11n/s/m/l/x.pt`, `yolo26n/s/m/l/x.pt`, YAML или путь |
| `--data` | из `config.json` | разовое переопределение пути к `data.yaml` |
| `--epochs` | `100` | число эпох |
| `--imgsz` | `640` | размер входа (кратно 32) |
| `--batch` | `16` | размер батча, `-1` — авто-подбор |
| `--device` | авто | `0`, `0,1`, `cpu`; пусто — GPU если есть |
| `--workers` | `8` | процессы загрузки данных (Windows: 2–4) |
| `--patience` | `50` | ранняя остановка, если mAP не растёт столько эпох |
| `--cache` | `none` | `ram` / `disk` — кэш изображений, ускоряет эпохи |
| `--seed` | `0` | зерно воспроизводимости |
| `--project` | `result/<датасет>/<модель>/runs` | каталог для прогонов (переопределяется редко) |
| `--name` | `train` | имя прогона внутри `--project` |
| `--exist-ok` | выкл. | писать в существующий каталог прогона |
| `--resume` | выкл. | продолжить прерванный прогон: ищет самый свежий **незавершённый** `weights/last.pt` по датасету; можно указать путь явно |

Полный список гиперпараметров Ultralytics (оптимизатор, LR, аугментации) —
в документации: https://docs.ultralytics.com/modes/train/#train-settings

### 4.6 Обучение YOLO26

**Модифицировать `train.py` не нужно** — достаточно сменить чекпоинт:

```bash
python train.py --model yolo26n.pt
```

Результаты попадут в `result/<датасет>/yolo26n/runs/train/` — отдельно от прогонов YOLO11.

Что действительно требуется:

| Требование | Детали |
|---|---|
| `ultralytics>=8.4.0` | YOLO26 появился в 8.4; на 8.3 скрипт упадёт при загрузке чекпоинта |
| формат датасета | тот же YOLO-детект (`data.yaml`, `nc`, `names`, txt-разметка) — менять ничего не надо |
| веса | `yolo26n/s/m/l/x.pt` (скачиваются `download_models.py --family yolo26`) |
| гиперпараметры | передавать не нужно: MuSGD, Progressive Loss и STAL включаются сами при обучении `yolo26*.pt` |

Отличия YOLO26, которые касаются не обучения, а запуска:

- **Две головы.** По умолчанию используется one-to-many (с NMS) — она точнее.
  Для инференса без NMS добавьте `nms=False` в `predict`/`val`/`export`:
  `yolo detect predict model="result/Garbage Detection.v8i.yolov11/yolo26n/runs/train/weights/best.pt" source=... nms=False`
- **Нет DFL** — проще экспорт в ONNX/TensorRT, быстрее на CPU.
- **`yolo26n-p2.yaml` / `yolo26-p6.yaml`** — только архитектуры без предобученных
  весов (мелкие объекты / большой вход); обучать с нуля, `--model yolo26n-p2.yaml`.

Проверка, что установленная версия умеет YOLO26:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python train.py --model yolo26n.pt --epochs 1 --imgsz 320 --batch 2 --name yolo26_smoke
```

---

## 5. Что получится после обучения

```
result/Road-damage.v3i.yolov11/yolo26n/runs/train/
├── weights/
│   ├── best.pt          # лучшая модель — её и используем
│   └── last.pt          # состояние последней эпохи (для --resume)
├── results.csv          # метрики по эпохам
├── results.png          # графики loss и метрик
├── confusion_matrix.png
├── labels.jpg           # распределение bbox по датасету
├── train_batch*.jpg     # примеры батчей с аугментациями
└── val_batch*.jpg       # предсказания на валидации
```

Путь собирается как `result/<имя каталога датасета>/<имя модели>/runs/<--name>`, поэтому
не смешиваются ни прогоны разных датасетов, ни эксперименты с разными моделями.
Второй запуск с тем же `--name` получит суффикс (`train2`), если не указан `--exist-ok`.

### 5.1 Про `--resume`

`--resume` продолжает **прерванный** прогон: Ultralytics восстанавливает оптимизатор,
номер эпохи и гиперпараметры из `last.pt`. После нормального завершения обучения
Ultralytics вычищает это состояние (`epoch = -1`, `optimizer = None`), и повторный
`resume=True` на таком файле молча начал бы обучение с нуля на другом датасете.
Поэтому `train.py` проверяет чекпоинт и в таком случае завершается с подсказкой:

```
[error] not a resumable checkpoint: .../yolo26s/runs/train/weights/last.pt
        the run is already finished (last.pt has no optimizer state).
        to fine-tune from it use: --model ".../weights/last.pt"
```

Итог: прервали Ctrl+C → `python train.py --resume`; обучение уже закончилось →
`python train.py --model .../weights/best.pt --name <новое имя>`.

---

## 6. Как понять, что модель обучилась хорошо

Смотрите `results.png` и `results.csv`, последние строки:

| Метрика | Смысл | Ориентир |
|---|---|---|
| `metrics/precision(B)` | доля верных срабатываний | > 0.8 |
| `metrics/recall(B)` | доля найденных объектов | > 0.7 |
| `metrics/mAP50(B)` | основная метрика качества | > 0.7 — рабочее качество |
| `metrics/mAP50-95(B)` | строгая метрика (точность рамки) | > 0.45 |

Признаки проблем:

- **train loss падает, val loss растёт** → переобучение: меньше эпох, больше аугментаций,
  модель поменьше (`yolo11n`).
- **обе кривые вышли на плато** → обычно достаточно; можно взять модель крупнее.
- **mAP почти 0** → проверить разметку и что `names` в `data.yaml` совпадает с классами в labels.

Оценка на отложенном тесте (после обучения):

```bash
yolo detect val model="result/Garbage Detection.v8i.yolov11/yolo11n/runs/train/weights/best.pt" data="datasets/Garbage Detection.v8i.yolov11/data.yaml" split=test
```

---

## 7. Проверка модели на своих изображениях

```bash
yolo detect predict model="result/Garbage Detection.v8i.yolov11/yolo11n/runs/train/weights/best.pt" source=path/to/images conf=0.25 save=True
```

Из Python:

```python
from ultralytics import YOLO

model = YOLO("result/Garbage Detection.v8i.yolov11/yolo11n/runs/train/weights/best.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
print(results[0].boxes.xyxy, results[0].boxes.conf, results[0].names)
```

Результаты складываются в `runs/detect/predict*/` (каталог Ultralytics по умолчанию;
чтобы держать их рядом с датасетом, добавьте `project=result/<датасет>/predict`).

---

## 8. Дообучение на новых данных

Добавили изображения в датасет → обновлять `data.yaml` не нужно (пути те же), просто
запустите обучение от лучших весов:

```bash
python train.py --model "result/Garbage Detection.v8i.yolov11/yolo11n/runs/train/weights/best.pt" --epochs 50 --name ft
```

Новый прогон попадёт в тот же каталог модели: `result/<датасет>/yolo11n/runs/ft/`
(имя модели берётся из пути чекпоинта, а не из имени файла `best`).

Отличие от `--resume`: `--model best.pt` начинает новый прогон с текущих весов
(можно менять гиперпараметры), `--resume` продолжает прерванный прогон с теми же настройками.

---

## 9. Экспорт модели

```bash
yolo export model="result/Garbage Detection.v8i.yolov11/yolo11n/runs/train/weights/best.pt" format=onnx imgsz=640
```

Форматы: `onnx`, `openvino`, `engine` (TensorRT), `torchscript`, `tflite` и др.

---

## 10. Перенос на Debian

1. Скопируйте каталог проекта **без** `.venv` и `result/` (окружение создаётся заново).
   Файлы `yolo11*.pt`, каталог `datasets/` и `config.json` скопируйте — тогда на сервере
   не понадобится интернет ни для весов, ни для данных.
2. `python3 -m venv .venv && source .venv/bin/activate`
3. Установите зависимости по `SETUP.md`.
4. Запустите `python train.py --device 0`.

Пути в `data.yaml` относительные, поэтому перенос каталога целиком ничего не ломает.
Если датасет лежит в другом месте — передайте `--data /path/to/data.yaml`.

---

## 11. Частые проблемы

| Симптом | Решение |
|---|---|
| `missing dependency: ultralytics` | активируйте `.venv` и выполните `pip install -r requirements.txt` |
| `config file is empty` / `no dataset path` | заполните `config.json`: `{"data": "datasets/<датасет>/data.yaml"}` |
| `data.yaml not found` | проверьте значение `data` в `config.json` или передайте `--data "datasets/.../data.yaml"` |
| результаты попали не в тот датасет | путь считается от `config.json`; посмотрите строки `[info] dataset` и `[info] model dir` в начале вывода |
| `not a resumable checkpoint` | прогон уже завершён; для дообучения используйте `--model .../weights/best.pt` |
| `CUDA out of memory` | уменьшите `--batch` (16 → 8 → 4), затем `--imgsz` (640 → 512) |
| обучение на CPU идёт очень долго | 4730 изображений на CPU — это часы-дни на эпоху-сотню; используйте GPU или уменьшите `--imgsz 320` для черновика |
| зависает на старте в Windows | `--workers 2`, иногда `--workers 0` |
| ошибка `DataLoader worker ... exited unexpectedly` | уменьшите `--workers` (Windows), проверьте свободную RAM |
| нужно прервать и продолжить | Ctrl+C, затем `python train.py --resume` |
| модель качается при старте | запустите `python download_models.py` заранее или укажите путь: `--model ./yolo11n.pt` |
| `Weights only load failed` / битый `.pt` | перекачайте: `python download_models.py --models n --force` |
