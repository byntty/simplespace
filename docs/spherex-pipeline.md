# SPHEREx: конвейер данных, QC и поиск новых объектов

**Практическое ТЗ и справочник для Нуржана.**
Актуально на ноябрь 2026. Все ключевые факты сверены с официальной документацией
Caltech/IPAC-IRSA и SPHEREx Explanatory Supplement; ссылки — в конце документа.
Версии продуктов, набор битов и имена колонок могут меняться между релизами,
поэтому код должен **читать определения из заголовков FITS**, а не полагаться
только на таблицы ниже.

---

## 0. Коротко: что должен делать конвейер

1. **Загрузить** Level 2 Spectral Image MEF (или cutout) для нужного участка неба.
2. **Проверить** шесть HDU и прочитать `IMAGE`, `FLAGS`, `VARIANCE`, `WCS-WAVE`,
   `PSF/EPSF`.
3. **Замаскировать** пиксели по битам `FLAGS` (космические лучи, насыщение,
   битые пиксели, артефакты), расширить маску на 1–2 пикселя.
4. **Привести** кадры к общей сетке WCS, вычесть фон, сматчить PSF и вычесть
   кадры разных эпох.
5. **Найти** источники в разностном изображении (SExtractor / photutils / SEP),
   отбросить всё, что совпало с маской или артефактом.
6. **Сверить** с каталогами WISE (AllWISE, CatWISE2020) и Gaia DR3;
   для Gaia учесть собственное движение на эпоху наблюдения.
7. **Классифицировать**: новый кандидат / известный объект / артефакт /
   транзиент. Требовать подтверждения как минимум в двух эпохах или каналах.
8. **Собрать спектр** по 102 каналам, сгруппировать соседние каналы по S/N,
   протащить флаги в результат.
9. **Опубликовать** для сайта компактный JSON-манифест (схема в §8):
   кандидаты, координаты, движение, флаги, маски, спектр, ссылки на PNG-превью.

Главное правило: **любой одиночный «горячий» пиксель, космический луч или
сатурация, не закрытые маской, при вычитании кадров выглядят как новый объект.**
Поэтому маскирование — не «улучшение», а обязательный этап.

---

## 1. Продукт и структура MEF

Основной продукт — Level 2 Spectral Image MEF, примерно 70 МБ на файл.
Один файл = **один детектор (D1–D6) и одна экспозиция**. Физически в файле
первичный HDU (метаданные, без данных) плюс **шесть расширений**:

| HDU | EXTNAME | Размер | Содержимое | Обязателен |
|----:|---------|--------|------------|------------|
| 0 | PRIMARY | — | служебные заголовки, версия пайплайна, калибровки | — |
| 1 | `IMAGE` | 2040×2040 float32 | поверхностная яркость, **MJy/sr**; зодиакальный свет **не вычтен** | **да** |
| 2 | `FLAGS` | 2040×2040 int32 | побитовая маска статусов пикселей | **да** |
| 3 | `VARIANCE` | 2040×2040 float32 | дисперсия потока, (MJy/sr)²; веса 1/σ² | крайне желательно |
| 4 | `ZODI` | 2040×2040 float32 | модель зодиакального фона, MJy/sr; **не вычтена** из IMAGE | желательно |
| 5 | `PSF` / `EPSF` | таблица/куб | PSF для позиции; в QR2 — куб 101×101×121, в QR3/DR1 — таблица ePSF | для фотометрии |
| 6 | `WCS-WAVE` | binary table | приблизительная привязка пиксель → λ, Δλ | для спектра/визуализации |

### Ключевые оговорки

- **`IMAGE` — это не готовый поток источника.** Единицы MJy/sr. Чтобы получить
  μJy, нужно умножить на карту телесного угла пикселя
  (`solid_angle_pixel_map_Dn_spx_cal-sapm-...fits`, arcsec²).
- **`ZODI` не вычтен.** Для поиска слабых объектов вычитайте локальный фон
  (SEP `Background`, кольца вокруг источника) или модель `ZODI`.
- **PSF зависит от длины волны и позиции.** В QR2 это куб 121 плоскость
  (11×11 зон), 101×101 на плоскость. В QR3/DR1 — `EPSF`: таблица ePSF
  (33×33, oversampling ×5), сетка 21×21 для D1, D2, D4, D5, D6 и 11×41
  для D3. Если `EPSF` нет — используйте `PSF`. Ошибка в описании PSF в
  заголовках QR2 до 31.03.2026 исправлена; не парсите старые заголовки вслепую.
- **`WCS-WAVE` — для визуализации.** Для науки берите `CWAVE`/`CBAND` из
  отдельного продукта `spectral_wcs_Dn_spx_cal-wcs-...fits`: это уже
  откалиброванные центральная длина волны и ширина полосы для каждого пикселя.
- **Пиксель 6.15″.** Это критично для радиуса кросс-матчинга и для того,
  что считать «смещением» объекта.
- Файл называет детектор: `level2_2025W18_2B_0001_1D2_spx_l2b-v12-2025-164.fits`
  → `D2` = detector 2.
- В релизе 6 MEF на одно наведение (по одному на детектор). Не все 6 полос
  обязаны присутствовать в архиве: QC оценивается по каждой полосе отдельно.

### Минимальный Python-каркас чтения

```python
from astropy.io import fits
from astropy.wcs import WCS

with fits.open(path, memmap=False) as hdul:
    hdul.info()
    image = hdul["IMAGE"].data.astype("float64")     # MJy/sr
    flags = hdul["FLAGS"].data.astype("int64")       # int32 → int64 для битов
    var   = hdul["VARIANCE"].data.astype("float64")
    hdr   = hdul["IMAGE"].header
    detector = int(hdr["DETECTOR"])                   # 1..6
    spatial_wcs = WCS(hdr)
    spectral_wcs = WCS(hdr, fobj=hdul, key="W")
    spectral_wcs.sip = None                            # SIP не относится к λ
    # PSF: в разных релизах разное имя
    psf_hdu = hdul["EPSF"] if "EPSF" in hdul else hdul["PSF"]
```

Проверки при загрузке:

- `hdul["IMAGE"].data.shape == (2040, 2040)`;
- `hdul["FLAGS"].data.shape` совпадает;
- `BUNIT` у `IMAGE` содержит `MJy/sr`;
- `DETECTOR` в 1..6;
- `WCS-WAVE` читается (`len(Table(hdul["WCS-WAVE"].data)) == 1`);
- `FLAGS` содержит ключи `MSKN*` / `MSKD*` (см. §2).

---

## 2. FLAGS: полная таблица битов

`FLAGS` — 32-битная побитовая маска (int32) на каждый пиксель. Значение —
сумма битов; проверять надо **побитово**, а не сравнивать число целиком
(кроме отладки). Определения продублированы в заголовке расширения:
`MSKNnnnn` — имя бита, `MSKDnnnn` — описание. **Стройте маски из заголовка**,
таблица ниже — для понимания и для значений по умолчанию.

| Бит | Имя | Маска (dec) | Что означает | Категория | Действие |
|---:|---|---:|---|---|---|
| 0 | `TRANSIENT` | 1 | транзиент/космический луч, обнаруженный на SUR | CR | маскировать + дилатация 2 px |
| 1 | `OVERFLOW` | 2 | пиксель близок к насыщению | насыщение | маскировать + дилатация |
| 2 | `SUR_ERROR` | 4 | ошибка onboard-обработки | CR/шум | маскировать |
| 6 | `NONFUNC` | 64 | постоянно нерабочий пиксель | битый | маскировать |
| 7 | `DICHROIC` | 128 | низкая эффективность из-за дихроика | битый/калибровка | маскировать, с осторожностью |
| 9 | `MISSING_DATA` | 512 | данные потеряны на борту | битый | маскировать |
| 10 | `HOT` | 1024 | горячий (шумный) пиксель | битый | маскировать |
| 11 | `COLD` | 2048 | аномально низкий сигнал | битый | маскировать |
| 12 | `FULLSAMPLE` | 4096 | доступна полная временная история | информация | **не маскировать** |
| 14 | `PHANMISS` | 16384 | коррекция фантома не применена | битый | маскировать |
| 15 | `NONLINEAR` | 32768 | нелинейность нельзя надёжно исправить | насыщение | маскировать |
| 17 | `PERSIST` | 131072 | персистентный заряд выше порога | битый | маскировать |
| 19 | `OUTLIER` | 524288 | выброс, найденный детектором выбросов | CR/шум | маскировать |
| 20 | `CROSSTALK` | 1048576 | перекрёстные помехи | артефакт | маскировать |
| 21 | `SOURCE` | 2097152 | пиксель сопоставлен с известным источником | источник | **не считать «битым»** |
| 22 | `GHOST` | 4194304 | призрак от яркого источника в экспозиции | артефакт | маскировать |
| 23 | `GHOST_FPA` | 8388608 | призрак от источника на другом детекторе | артефакт | в L2-изображениях обычно нет |
| 24 | `GHOST_EXT` | 16777216 | призрак от источника вне поля | артефакт | маскировать |
| 25 | `STREAK` | 33554432 | полоса/трек | артефакт | маскировать |
| 26 | `BLOOM` | 67108864 | блуминг от насыщения | насыщение | маскировать |
| 27 | `SNOWBALL` | 134217728 | снежок (частица/блик) | артефакт | маскировать |
| 28 | `HALO` | 268435456 | гало вокруг транзиента | CR-артефакт | маскировать |
| 29 | `SATELLITE_HALO` | 536870912 | гало/трек от спутника | артефакт | маскировать |
| 32 | `CONTAINS_BAD_PIXEL` | 4294967296 | в измерении был плохой пиксель (derived) | QC фотометрии | отбраковка/флаг |
| 33 | `FIT_ERROR` | 8589934592 | ошибка сходимости фотометрии (derived) | QC фотометрии | отбраковка |

Биты 3–5, 8, 13, 30–31 в текущей публичной таблице IRSA не описаны
(резерв/продукты, не входящие в публичный релиз). В старых версиях
документации встречались `VAR_UNDER` (16) и `PERSIST_UNK` (18); в актуальной
таблице их нет. Не хардкодьте их: читайте `MSKNnnnn` из `FLAGS`-заголовка.

Биты 32–33 не помещаются в signed int32 и относятся к выходу
спектрофотометрии/форсированной фотометрии, а не к массиву `FLAGS` изображения.
Для них используйте int64. Значение `flags` в выходе инструмента может быть
очень большим; «большое число» — это не «плохой пиксель», а набор битов.

### Рекомендуемые маски

```python
# библиотека масок; значения — текущий публичный набор
TRANSIENT_BITS = (1 << 0) | (1 << 2) | (1 << 19) | (1 << 28) | (1 << 29)
SATURATION_BITS = (1 << 1) | (1 << 15) | (1 << 26)
BAD_PIXEL_BITS = ((1 << 6) | (1 << 7) | (1 << 9) | (1 << 10) |
                  (1 << 11) | (1 << 14) | (1 << 17))
ARTIFACT_BITS = ((1 << 20) | (1 << 22) | (1 << 23) | (1 << 24) |
                 (1 << 25) | (1 << 27))
SOURCE_BIT = 1 << 21
FULLSAMPLE_BIT = 1 << 12

BAD_FOR_DETECTION = (TRANSIENT_BITS | SATURATION_BITS |
                     BAD_PIXEL_BITS | ARTIFACT_BITS)
```

Замечания:

- **`SOURCE` не входит в «плохие» пиксели.** Это пиксели известных источников.
  Для оценки фона их полезно исключать, для поиска переменности — нет:
  именно там и бывают реальные транзиенты. В difference imaging не маскируйте
  `SOURCE` глобально; вместо этого сопоставляйте кандидатов с каталогом.
- **`FULLSAMPLE` — информация, не дефект.** Не включайте в маску.
- **Дилатация.** `TRANSIENT`, `BLOOM`, `HALO`, `STREAK` расширяйте на 1–3 px:
  крыло космического луча и блуминг выходят за границы флага. Для насыщения
  маскируйте весь блум-«стебель», а не только центральный пиксель.
- **Границы и дихроик.** У краёв детектора и в области D3/D4 возможны
  систематические эффекты; `DICHROIC` требует отдельной осторожности в
  калибровке, а не просто «замазать и забыть».
- В официальном Source Discovery notebook для быстрой демонстрации оставляют
  «хорошими» только `flags == 0`, `2` (`SUR_ERROR`) и `2097152` (`SOURCE`).
  Это **демо-уровень**, а не строгий QC: такая маска пропускает часть
  `OUTLIER`, `BLOOM`, `HALO`, `PERSIST` и других артефактов. Для поиска
  новых объектов берите полный `BAD_FOR_DETECTION`.

### Проверка маски и метрики

- Считайте долю пикселей под маской по каждой группе; для реального кадра
  сверяйтесь с заголовочными счётчиками `HIERARCH L2 N_*` (`N_TRANSIENT`,
  `N_OVERFLOW`, `N_NONFUNC`, ...). Они дают число пикселей с флагом.
- Для апертурной фотометрии требуйте, чтобы в апертуре не было пикселей
  `CONTAINS_BAD_PIXEL`; иначе ставьте флаг качества, а не «0 ошибок».
- Храните в результате **какие биты были замаскированы** и их значения —
  это нужно для воспроизводимости и для интерфейса.

---

## 3. Привязка кадра к полосе, детектору и каналу

### 3.1. Шесть детекторов = шесть полос

SPHEREx покрывает 0.75–5.00 мкм шестью детекторами. Каждый детектор — это
одна спектральная полоса (band) и 17 каналов (channels), всего **6 × 17 = 102**.

| Детектор | Полоса | λ, мкм | R = λ/Δλ | Каналов |
|---:|---|---|---:|---:|
| D1 | 1 | 0.75–1.09 | 39 | 17 |
| D2 | 2 | 1.10–1.62 | 41 | 17 |
| D3 | 3 | 1.63–2.41 | 41 | 17 |
| D4 | 4 | 2.42–3.82 | 35 | 17 |
| D5 | 5 | 3.83–4.41 | 112 | 17 |
| D6 | 6 | 4.42–5.00 | 128 | 17 |

Диапазоны — официальные границы полос; конкретный канал внутри полосы уже
(обычно Δλ ≈ λ/R). Границы D3/D4 (2.41/2.42 мкм) — дихроик, к нему нужна
отдельная осторожность при калибровке.

Пары **D1/D4, D2/D5, D3/D6** снимаются одновременно (один и тот же участок
неба в один момент). Это важно для difference imaging: для поиска переменности
сравнивайте одну и ту же полосу в разные эпохи; для цветовой диагностики в
одну эпоху можно использовать парные детекторы, но это будет разная λ, а не
«та же картинка».

### 3.2. Где это в заголовках

- `DETECTOR` (int, 1–6) в заголовке `IMAGE` — **основная привязка**.
- Имя файла содержит `D1`…`D6` — например, `..._1D2_spx_l2b-...fits`.
- `OBSID`, `DATE-OBS` / `DATE`, `PSF_FWHM` — идентификатор и условия экспозиции.
- `BUNIT = 'MJy/sr'` — единицы `IMAGE`.
- В TAP/SIA полоса передаётся как `energy_bandpassname = 'SPHEREx-D1'` … `'SPHEREx-D6'`.
- `WCS-WAVE` в MEF — приблизительная λ; **не заменяет** калиброванный продукт.

### 3.3. 102 полосы и номер канала

Геометрический номер полосы:

```python
band_index = (detector - 1) * 17 + channel   # 1..102
# обратно:
detector = (band_index - 1) // 17 + 1        # 1..6
channel  = (band_index - 1) % 17 + 1         # 1..17
```

Карта каналов (`spectral_channels_spx_cal-sch-...fits`) содержит целые
1–17; значения **0 и 18** означают «пиксель вне каналов» (края, промежутки).
Не считайте 0 валидным каналом и не подставляйте его в формулу.

Важно: 17 каналов — это **не 17 равных срезов** полосы. Их центры и ширины
заданы калибровкой. Для науки берите `CWAVE`/`CBAND` из
`spectral_wcs_Dn_spx_cal-wcs-...fits` (расширения `CWAVE`, `CBAND`), а не
линейную интерполяцию «0.75 + ...».

```python
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

# 1) геометрия кадра
with fits.open(mef_path) as hdul:
    det = int(hdul["IMAGE"].header["DETECTOR"])
    ch_map = fits.getdata(channel_map_path)  # int, 0..18
    valid = np.isin(ch_map, np.arange(1, 18))

# 2) калиброванная длина волны из spectral_wcs
with fits.open(spectral_wcs_path) as hdul:
    cwave = hdul["CWAVE"].data   # мкм, 2040x2040
    cband = hdul["CBAND"].data   # мкм, ширина
    hdr = hdul["CWAVE"].header

# 3) индекс полосы для пикселя с каналом ch
ch = int(ch_map[y, x])
assert 1 <= ch <= 17, "pixel outside the 17 science channels"
band_index = (det - 1) * 17 + ch
lam = float(cwave[y, x])

# 4) WCS-WAVE — только для быстрой визуализации
with fits.open(mef_path) as hdul:
    wwave = WCS(hdul["IMAGE"].header, fobj=hdul, key="W")
    wwave.sip = None
    lam_approx, _ = wwave.pixel_to_world_values(x, y)
```

Если нужен «один кадр — одна полоса», это `Dn` + номер канала; если нужен
спектр — берите все 17 каналов и стройте λ по `CWAVE`.

### 3.4. Группировка соседних длин волн: что она даёт и чего не даёт

| Задача | Группировка помогает? | Почему |
|---|---|---|
| Увеличить S/N слабого источника | **да** | шум падает как √N при объединении независимых каналов |
| Построить устойчивый спектр | **да** | адаптивное биннирование по S/N, меньше пустых бинов |
| Повысить угловое разрешение | **нет** | разрешение задаётся дифракцией и пикселем 6.15″, а не числом каналов |
| «Сделать картинку чётче» | **нет** | коадд расширяет эффективную полосу и размывает цветовую структуру |

SPHEREx — низкое угловое разрешение: PSF FWHM ≈ 5–8″ при пикселе 6.15″.
Коаддинг соседних каналов **не делает изображение резче**. Если нужна
«чёткость», работайте с ePSF: деконволюция/PSF-фитинг, а не сумма каналов.
Для детекции группируйте каналы внутри одной полосы (PSF меняется слабо),
но не смешивайте D2 и D4: получите широкую эффективную полосу и потеряете
спектральную информацию.

Практическое правило: для детекции — 2–4 соседних канала или адаптивные
бины; для фотометрии — фиксированная апертура в единицах PSF; для спектра —
бины с S/N ≥ 3–5 и обязательным сохранением исходных флагов.

---

## 4. Доступ к данным и автообновление

### 4.1. Коллекции и сервисы

| Ресурс | Значение |
|---|---|
| SIA2 | `https://irsa.ipac.caltech.edu/SIA` |
| Коллекции DR1 | `spherex_dr1`, `spherex_dr1_deep` |
| Коллекции QR3 | `spherex_qr3`, `spherex_qr3_deep` |
| TAP | `https://irsa.ipac.caltech.edu/TAP` |
| Cutout | `https://irsa.ipac.caltech.edu/` + `artifact.uri` + `?center=RA,Decd&size=SIZE` |
| Документация | `https://caltech-ipac.github.io/spherex-archive-documentation/` |

Пример SIA2-запроса (PyVO):

```python
import pyvo

sia = pyvo.dal.SIAService("https://irsa.ipac.caltech.edu/SIA")
rows = sia.search(
    pos=(ra, dec), size=0.1,   # градусы
    collection="spherex_dr1",
    band="SPHEREx-D2",         # energy_bandpassname
).to_table()
# rows['access_url'], rows['energy_bandpassname'], rows['obs_id']
```

Точные имена параметров SIA зависят от версии PyVO; если `band` не
принимается, фильтруйте таблицу по `energy_bandpassname` после запроса.

### 4.2. Cutout: не тяните 70 МБ на каждую проверку

Официальный ноутбук `spherex_cutouts` берёт `artifact.uri` из TAP и добавляет
параметры cutout. Запрос (подставьте свои RA/Dec/размер):

```sql
SELECT 'https://irsa.ipac.caltech.edu/' || a.uri
       || '?center=210.80227,54.34895d&size=0.05' AS uri,
       p.time_bounds_lower,
       p.energy_bandpassname
FROM spherex.artifact a
JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', 210.80227, 54.34895), p.poly)
ORDER BY p.time_bounds_lower
```

В ADQL строковая конкатенация и геометрия работают; параметры подставляйте
числами/строками, а не `?`-bind, если сервис не поддерживает bind в ADQL.
Cutout отдаёт FITS с теми же расширениями (IMAGE/FLAGS/…), но меньшего
размера; для difference imaging берите cutout с запасом ≥ 2–3 PSF.

### 4.3. Как ловить «новые данные»

1. Храните `last_ingest_utc` и список уже обработанных `obsid`/`planeid`.
2. Запрашивайте планы с `time_bounds_lower > last_ingest_utc`:

```sql
SELECT planeid, obsid, energy_bandpassname, time_bounds_lower, time_bounds_upper, poly
FROM spherex.plane
WHERE time_bounds_lower > 2026-09-01T00:00:00Z
  AND energy_bandpassname IN ('SPHEREx-D2')
ORDER BY time_bounds_lower
```

3. Для каждого нового плана проверяйте, что он пересекает поле и что
   `DETECTOR` совпадает с нужной полосой.
4. Скачивайте cutout/MEF, прогоняйте QC; **не публикуйте** объект, пока он
   не подтверждён минимум двумя эпохами (или эпохой + независимым каналом).
5. Обновляйте манифест атомарно: сначала временный файл, потом rename.
6. Дедуплицируйте кандидатов по координатам и `obsid`; не создавайте нового
   кандидата на каждый cutout одного и того же объекта.

Полные MEF качать в браузере нельзя: 70 МБ × 6 × эпохи, плюс CORS и
отсутствие FITS-ридера. Нужен backend (Python), который отдаёт сайту только
JSON и PNG. Это архитектурное ограничение, а не «недостаток сайта».

---

## 5. Официальные ноутбуки IRSA: что брать

Репозиторий: `https://github.com/Caltech-IPAC/irsa-tutorials`.
Опубликованные версии:

- `https://caltech-ipac.github.io/irsa-tutorials/spherex-intro`
- `https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts`
- `https://caltech-ipac.github.io/irsa-tutorials/spherex-psf`
- `https://caltech-ipac.github.io/irsa-tutorials/spherex-source-discovery-tool-demo`

### 5.1. Intro

Что взять: имена HDU, `DETECTOR`, единицы `MJy/sr`, структуру `WCS-WAVE`,
работу с `FLAGS`, различие `PSF`/`EPSF` между релизами. Не копируйте
жёстко зашитые пути к файлам — стройте их из `obsid`/`detector`.

### 5.2. Cutouts

Что взять: TAP-запрос `artifact JOIN plane`, формирование cutout-URL,
скачивание FITS. Это основа для «подгрузить только поле вокруг кандидата».

### 5.3. PSF

Что взять: чтение `EPSF`, выбор ePSF по позиции, oversampling ×5,
нормировку ядра. Для фотометрии и PSF-matching используйте ePSF, а не
гауссову аппроксимацию.

### 5.4. Source Discovery Tool

Что взять:

- фон: `sep.Background(image, bw=64, bh=64, fw=11, fh=11)`, вычитание `bkg.back()`;
- репроекция: `reproject_exact` для потока, `reproject_interp` для маски/сегментации;
- детекция: SExtractor (`DETECT_THRESH=5`, `DETECT_MINAREA=5` — проверяйте под свои данные);
- фотометрия: апертуры 3/4/5 px, поправка на апертуру, перевод MJy/sr → μJy
  через карту телесного угла;
- отбраковка: список плохих флагов, проверка краёв, `fit_ql`.

Важно: в ноутбуке вычитаются **разные полосы** (например, D2 − D4) для
поиска красных источников, а не две эпохи. Для «новых объектов после
обновления» нужен **временной** difference imaging: та же полоса, разные
`time_bounds_lower`. Не путайте эти задачи.

Минимальный рабочий фрагмент (адаптированный, не дословная копия):

```python
import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact

with fits.open(mef_a) as a, fits.open(mef_b) as b:
    img_a, hdr_a = a["IMAGE"].data.astype("f8"), a["IMAGE"].header
    img_b, hdr_b = b["IMAGE"].data.astype("f8"), b["IMAGE"].header
    flg_a = a["FLAGS"].data.astype("i8")
    flg_b = b["FLAGS"].data.astype("i8")

bad = build_bad_mask(flg_a) | build_bad_mask(flg_b)  # §2
bkg = sep.Background(img_a, mask=bad, bw=64, bh=64, fw=11, fh=11)
img_a = img_a - bkg.back()

# B → сетка A, с сохранением потока
img_b_on_a, _ = reproject_exact((img_b, WCS(hdr_b)), WCS(hdr_a), shape_out=img_a.shape)

diff = img_a - img_b_on_a
diff[bad] = 0.0
```

### 5.5. Spectrophotometry Tool

Веб-инструмент IRSA встроен в **SPHEREx Data Explorer** (раздел
Spectrophotometry Tool); отдельного публичного endpoint нет. Документация:
`https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/sp.html`.
Это **форсированная фотометрия** по позиции: Tractor фитирует источник по
всем каналам, а не «заново ищет» его. Что важно для конвейера:

- на вход — координаты и радиус/позиция; на выход — спектр по каналам;
- колонки результата включают `flags`, `fit_ql`, `flux_bkg`, длину волны и поток;
- `flags` — это уже производные биты (в т.ч. 32 `CONTAINS_BAD_PIXEL`,
  33 `FIT_ERROR`); `fit_ql` — качество фита, а не «0/1»;
- не игнорируйте `DICHROIC` (7) и `FIT_ERROR` (33): на краях полос и при
  плохом PSF фотометрия смещена;
- для новых объектов сначала нужна позиция из difference imaging, потом
  форсированная фотометрия по этой позиции.

Ноутбука Spectrophotometry Tool в публичном `irsa-tutorials` может не быть;
интерфейс и выходные колонки сверяйте с online help:
`https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/sp.html`.

---

## 6. Кросс-матчинг с WISE и Gaia

### 6.1. Где что лежит (важная поправка)

| Каталог | Сервис | TAP endpoint | Таблица |
|---|---|---|---|
| AllWISE | IRSA | `https://irsa.ipac.caltech.edu/TAP` | `allwise_p3as_psd` |
| CatWISE2020 | IRSA | `https://irsa.ipac.caltech.edu/TAP` | `catwise_2020` |
| Gaia DR3 | **ESA** | `https://gea.esac.esa.int/tap-server/tap` | `gaiadr3.gaia_source` |

**Gaia нет в IRSA TAP.** IRSA — инфракрасный архив NASA; Gaia обслуживает
ESA. Один SQL-запрос «ко всем трём каталогам сразу» в IRSA не пройдёт.
Рабочие варианты:

1. Два TAP-запроса (IRSA для WISE, ESA для Gaia) и локальный кросс-матч
   через `astropy.coordinates` — рекомендуется.
2. Загрузить список координат в Gaia TAP как upload-таблицу и сделать
   кросс-матч на стороне ESA (`TAP_UPLOAD`), но это сложнее и не нужно
   для полей масштаба cutout.

### 6.2. ADQL: AllWISE

```sql
SELECT TOP 1000 designation, ra, dec, sigra, sigdec,
       w1mpro, w1sigmpro, w2mpro, w2sigmpro, w3mpro, w4mpro,
       ph_qual, cc_flags, ext_flg
FROM allwise_p3as_psd
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', 210.80227, 54.34895, 0.05))
```

`ph_qual` — качество фотометрии по W1–W4 (A/B/C/U); для надёжного
отождествления требуйте A/B в W1/W2. `ext_flg` показывает протяжённость;
`cc_flags` — артефакты/смешения. Не считайте любой WISE-объект «звездой»:
при 6.1″ WISE смешивает звёзды и галактики.

### 6.3. ADQL: CatWISE2020

```sql
SELECT TOP 1000 source_name, ra, dec, sigra, sigdec,
       w1mpro, w1sigmpro, w2mpro, w2sigmpro,
       pmra, pmdec, sigpmra, sigpmdec, snr1, snr2, ph_qual
FROM catwise_2020
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', 210.80227, 54.34895, 0.05))
  AND snr1 > 5
```

CatWISE лучше AllWISE для слабых и движущихся объектов: есть `pmra`/`pmdec`.
Точные имена колонок зависят от версии таблицы; проверьте:

```sql
SELECT column_name, datatype, description
FROM TAP_SCHEMA.columns
WHERE table_name = 'catwise_2020'
```

### 6.4. ADQL: Gaia DR3

```sql
SELECT TOP 1000 source_id, ra, dec, pmra, pmdec,
       parallax, phot_g_mean_mag, ruwe, astrometric_excess_noise
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', 210.80227, 54.34895, 0.05))
  AND phot_g_mean_mag < 20.5
```

Gaia-эпоха — **J2016.0**. `pmra` — это μ_α* = dα/dt·cosδ (mas/год).
Перед матчингом с SPHEREx обязательно приведите координаты к эпохе
наблюдения, иначе быстрая звезда «сместится» и вы получите ложный новый
кандидат.

### 6.5. Python: два TAP и локальный матч

```python
import numpy as np
import pyvo
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy import units as u

RA, DEC, RADIUS_DEG = 210.80227, 54.34895, 0.05

irsa = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
wise_adql = f"""
SELECT designation, ra, dec, w1mpro, w2mpro, ph_qual, ext_flg
FROM allwise_p3as_psd
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {RA}, {DEC}, {RADIUS_DEG}))
"""
wise = irsa.search(wise_adql).to_table()

from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = 2000
gaia_adql = f"""
SELECT source_id, ra, dec, pmra, pmdec, parallax,
       phot_g_mean_mag, ruwe, astrometric_excess_noise
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {RA}, {DEC}, {RADIUS_DEG}))
  AND phot_g_mean_mag < 20.5
"""
gaia = Gaia.launch_job(gaia_adql).get_results().to_pandas()
```

Приведение Gaia на эпоху SPHEREx (без параллакса/радиальной скорости —
приближение для малых углов; для быстрых близких звёзд лучше
`SkyCoord.apply_space_motion` с расстоянием):

```python
def gaia_at_epoch(ra, dec, pmra_mas_yr, pmdec_mas_yr, mjd_obs, epoch_ref=2016.0):
    dt = mjd_obs / 365.25 + 2000.0 - epoch_ref   # MJD → десятичный год
    dra = pmra_mas_yr * dt / 3.6e6 / np.cos(np.deg2rad(dec))
    ddec = pmdec_mas_yr * dt / 3.6e6
    return ra + dra, dec + ddec

# если есть parallax и radial_velocity — точнее через SkyCoord:
# c = SkyCoord(ra=..., dec=..., distance=..., pm_ra_cosdec=...,
#              pm_dec=..., radial_velocity=..., obstime=Time("J2016.0"))
# c_obs = c.apply_space_motion(new_obstime=Time(mjd_obs, format="mjd"))
```

Матчинг:

```python
targets = SkyCoord(ra=ra_candidates*u.deg, dec=dec_candidates*u.deg)
ref = SkyCoord(ra=gaia["ra"].values*u.deg, dec=gaia["dec"].values*u.deg)
idx, sep2d, _ = match_coordinates_sky(targets, ref)
sep_arcsec = sep2d.arcsec
matched = sep_arcsec < 2.5   # для Gaia; для WISE — 5–6″
```

Радиусы и логика:

- **Gaia:** 2–3″ (суб-арксекундная точность, но у SPHEREx центроид хуже).
- **WISE:** 5–6″ (луч 6.1″ на W1; иначе смешаете соседние источники).
- **CatWISE/AllWISE:** для протяжённых галактик — 6–8″ с проверкой `ext_flg`.
- **Движущиеся объекты:** после PM-коррекции те же радиусы; для объектов
  Солнечной системы нужен эфемеридный матч (не покрыто этим документом).
- **Новый кандидат:** нет ни Gaia, ни WISE в радиусе после PM-коррекции,
  но есть значимая детекция в difference imaging. «Нет Gaia» само по себе
  ничего не значит: Gaia слабее/хуже видит красные пылевые источники.
  «Нет WISE» тоже: WISE не видит очень слабые/голубые объекты.

### 6.6. Качество каталогов

- AllWISE: `ph_qual` A/B, `snr` W1/W2 ≥ 5, `cc_flags` = 0 для чистых
  фотометрических измерений; `ext_flg` > 0 — протяжённый источник.
- CatWISE2020: `snr1`/`snr2` ≥ 5; PM только для объектов с надёжным
  многолетним покрытием; есть смешение в плотных полях.
- Gaia: `ruwe < 1.4` (хорошая астрометрия); `astrometric_excess_noise` —
  признак двойственности/плохого решения; **не отбрасывайте высокий PM** —
  это как раз кандидаты в быстрые звёзды, но корректируйте эпоху.
- Кросс-матч всегда храните с радиусом, эпохой и версией каталога:
  без этого результат невоспроизводим.

---

## 7. Алгоритм поиска новых объектов

### 7.1. Последовательность

```text
[1] INGEST      выбрать Dn и эпохи; проверить HDUs/WCS/единицы
[2] MASK        FLAGS → bad_pixel_mask; дилатация 1–3 px
[3] PHOTOMETRY  MJy/sr → μJy через solid angle; фон SEP/ZODI
[4] ALIGN       reproject_exact на общий WCS; residual < 0.5 px
[5] PSF-MATCH   ePSF; сверка FWHM/формы; при необходимости — ядро свёртки
[6] DIFFERENCE  epoch B − epoch A; шум из VARIANCE; порог 5σ
[7] DETECT      SExtractor/SEP; minarea ≥ 5; edge/negative-lobe veto
[8] VETO        флаги в centroid; CR/ghost/saturation/streak/persistence
[9] CATALOG     Gaia (PM-corrected) + WISE; новый = нет совпадений
[10] MULTI-EPOCH подтверждение в ≥2 эпохах (или эпоха + 2 канала)
[11] SPECTRUM    forced photometry по 102 каналам; бины по S/N
[12] PUBLISH     JSON-манифест + PNG; ручная проверка
```

### 7.2. Difference imaging: детали, которые ломают результат

- **Сравнивайте одну и ту же полосу.** D2 с D2, D4 с D4. D2 − D4 — это
  цвет, а не переменность; так можно найти красный источник, но не «новый
  объект после обновления».
- **Выравнивание.** Ошибка WCS даже в 1 px даёт дипольную пару
  «яркое/тёмное» и ложный кандидат. Проверяйте остаточный сдвиг по ярким
  звёздам; при необходимости `tweakreg`/фазовая кросс-корреляция.
- **PSF.** Разные эпохи — слегка разные PSF. Без PSF-matching остаются
  кольца вокруг ярких звёзд. Используйте ePSF; для быстрой проверки —
  гауссово ядро с отношением FWHM, но не выдавайте это за фотометрию.
- **Фон.** Локальный фон меняется (зодиакальный свет, рассеянный свет).
  Вычитайте сигма-клиппованный фон по каждой эпохе; `ZODI` — модель, а не
  измерение.
- **Шум.** Взвешивайте по `VARIANCE`; в разности дисперсии складываются.
  Не используйте глобальный шум по всему кадру — в поле есть градиенты.
- **Космические лучи.** `TRANSIENT`/`SUR_ERROR` + дилатация. Одиночный
  горячий пиксель без флага — проверять по `OUTLIER`, размеру и отсутствию
  PSF-профиля.
- **Насыщение.** `OVERFLOW`/`NONLINEAR`/`BLOOM`; маскируйте весь блум и
  окрестность, иначе разность даст «новый» яркий объект.
- **Призраки и streak.** `GHOST*`, `STREAK`, `SATELLITE_HALO`; они могут
  двигаться вместе с яркой звездой и имитировать транзиент.

### 7.3. Отбор кандидатов (практичные пороги)

| Критерий | Рекомендация |
|---|---|
| Пик разности | ≥ 5σ локального шума |
| Площадь | ≥ 5 px над 3σ (после PSF) |
| Форма | согласуется с ePSF; `elongation` не экстремальная |
| Негативный лепесток | |negative| / |positive| < 0.3 |
| Флаги | 0 плохих битов в centroid и апертуре |
| Край | ≥ 1.5 × PSF FWHM от края |
| Каталог | нет Gaia/WISE в радиусе после PM |
| Эпохи | ≥ 2 независимых эпохи (или 2 канала с согласованным спектром) |
| Движение | для PM-кандидата — согласованный сдвиг между эпохами |
| Артефакты | не CR/ghost/streak/persistence по флагам и морфологии |

Скор кандидата (пример, не «истина»):

```python
score = (0.35 * min(snr / 10, 1.0) +
         0.20 * min(n_epochs / 3, 1.0) +
         0.20 * (1.0 if not catalog_match else 0.0) +
         0.15 * psf_consistency +          # 0..1
         0.10 * (1.0 - min(bad_fraction / 0.2, 1.0)))
```

Классы результата:

- `new_candidate` — значимая разность, нет каталога, ≥2 эпохи;
- `known_star` — Gaia-совпадение после PM;
- `known_galaxy` — WISE/AllWISE, нет Gaia, протяжённая морфология;
- `artifact` — флаг/CR/ghost/streak/насыщение;
- `transient` — есть, но пока одна эпоха; в публикацию «новых» не идёт;
- `rejected` — негативный лепесток, край, плохой fit.

### 7.4. Спектр и 102 канала

1. Для позиции кандидата сделайте форсированную фотометрию по каждому
   каналу (или возьмите `spectral_channels` + апертуру).
2. Переведите MJy/sr → μJy: `flux_ujy = image * solid_angle_arcsec2 * 1e6`
   (1 MJy/sr · arcsec² = 1e6 μJy, если телесный угол в arcsec²).
3. Сгруппируйте соседние каналы до S/N ≥ 3–5; сохраните исходные
   `CWAVE`/`CBAND` и флаги каждого канала.
4. Не сглаживайте через границу D3/D4 и через дихроик без отметки.
5. Если канал имеет `CONTAINS_BAD_PIXEL`/`FIT_ERROR` — не выбрасывайте молча,
   помечайте `null`/флагом и показывайте в UI.

### 7.5. Чего избегать

-Не считать «новый объект» по одному кадру без маски и без второй эпохи.
- Не маскировать `SOURCE` глобально: потеряете переменные звёзды.
- Не сравнивать каталог без PM-коррекции: быстрые звёзды дадут ложные «смещения».
- Не использовать WISE-радиус 1–2″: луч 6.1″, вы просто не найдёте совпадение.
- Не смешивать «нет каталога» и «объект новый»: каталоги неполны.
- Не публиковать кандидата без указания версии релиза, эпохи, полосы и маски.

---

## 8. Контракт данных для сайта

Сайт не должен читать FITS. Backend Нуржана делает всю тяжёлую работу и
отдаёт один компактный JSON + PNG-превью. Ниже — минимальная схема,
которую уже понимает демо-адаптер (в `observatory.html` данные процедурные,
но поля совпадают по смыслу).

```json
{
  "schema": 1,
  "release": "DR1-2026.45",
  "generated_at": "2026-11-14T12:00:00Z",
  "demo": false,
  "fields": [
    {
      "id": "SPX-210.8023+54.3490",
      "ra": 210.80227,
      "dec": 54.34895,
      "product": {
        "obs_id": "2025W18_2B_0001_1",
        "detector": 2,
        "band": 2,
        "channel": 7,
        "band_index": 24,
        "lambda_um": 1.312,
        "hdus": ["IMAGE", "FLAGS", "VARIANCE", "ZODI", "EPSF", "WCS-WAVE"],
        "image_url": "/previews/spx-210.8023+54.3490.png",
        "mask_url": "/previews/spx-210.8023+54.3490-mask.png",
        "cutout_url": "https://irsa.ipac.caltech.edu/..."
      },
      "qc": {
        "masked_fraction": 0.043,
        "bit_counts": {"TRANSIENT": 120, "OVERFLOW": 33, "NONFUNC": 87},
        "bits_used": [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19, 20, 22, 24, 25, 26, 27, 28, 29]
      },
      "catalog": {
        "match_radius_arcsec": 3.0,
        "wise": [
          {"name": "J140520.54+542055.7", "ra": 210.8356, "dec": 54.3488, "w1": 12.1, "w2": 11.9, "quality": "AA"}
        ],
        "gaia": [
          {"source_id": "1234567890", "ra": 210.8021, "dec": 54.3491, "g": 15.2, "pm_mas_yr": [12.4, -8.1]}
        ]
      },
      "candidates": [
        {
          "id": "CAND-001",
          "ra": 210.80227,
          "dec": 54.34895,
          "status": "new_candidate",
          "snr": 8.4,
          "epochs": 2,
          "flags": [],
          "catalog_match": null,
          "motion_arcsec": 0.0,
          "score": 0.82
        }
      ],
      "spectrum": {
        "wave_um": [1.25, 1.30, 1.35],
        "flux_ujy": [12.1, 14.0, null],
        "flags": ["OK", "OK", "FIT_ERROR"]
      }
    }
  ]
}
```

Правила контракта:

- `release` и `generated_at` обязательны; без них нельзя отличить новые данные.
- `demo: true` — только для процедурной демонстрации; сайт обязан показывать
  метку `SIMULATED`/`DEMO FEED`.
- `hdus` — фактически присутствующие расширения; если нет `FLAGS`, кандидаты
  не публикуются (нельзя гарантировать маску).
- `qc.bit_counts` — из `FLAGS`-заголовка, не «на глаз».
- `catalog` — только качественные совпадения; храните радиус и эпоху.
- `candidates[].status`: `new_candidate` | `known_star` | `known_galaxy` |
  `artifact` | `transient` | `rejected`.
- `spectrum.flux_ujy` может быть `null` при плохом канале; `flags` обязательны.
- никаких FITS, токенов и путей к локальным дискам в JSON.

Сайт уже готов принять такой манифест: замените процедурный `SURVEY`/
`buildProduct()` на `fetch('/api/feed')` и сохраните те же поля. Пока backend
нет — интерфейс работает как демонстрация контракта, а не как «live NASA».

---

## 9. Чек-лист Нуржану

### 9.1. Юнит-тесты (без реальных данных)

- [ ] `decode_flags(1)` → `TRANSIENT`; `decode_flags(1 | (1 << 21))` →
      `TRANSIENT`, `SOURCE`; значение 0 — чистый пиксель.
- [ ] `band_index(1, 1) == 1`; `(1, 17) == 17`; `(2, 1) == 18`;
      `(6, 17) == 102`; канал 0/18 — `ValueError`.
- [ ] маска: синтетический CR с флагом `TRANSIENT` не попадает в детекцию
      после дилатации; без маски — попадает (контрольный тест).
- [ ] int64: биты 32/33 не теряются при OR.
- [ ] `gaia_at_epoch`: звезда с PM 100 mas/год за 10 лет сдвигается на ~1″
      (с учётом cosδ).
- [ ] JSON валидируется схемой; `demo`/`release` присутствуют.

### 9.2. Интеграционные проверки

- [ ] Открыть реальный MEF: 6 расширений, формы 2040×2040, `BUNIT`.
- [ ] Проверить `DETECTOR`/`Dn`/`energy_bandpassname` на трёх разных файлах.
- [ ] Скачать cutout через TAP-запрос; убедиться, что WCS не потерян.
- [ ] Прогнать одну пару эпох D2: маска → репроекция → разность → детекция.
- [ ] Сверить 5 известных звёзд с Gaia после PM; 5 галактик с WISE.
- [ ] Инжектировать искусственный источник в разность — пайплайн должен
      найти его с теми же параметрами.
- [ ] Одиночный CR и насыщение не должны попасть в `new_candidate`.
- [ ] Спектр: 17 каналов, λ из `CWAVE`, флаги не потеряны.

### 9.3. Эксплуатация

- [ ] Логи: `release`, `obsid`, `detector`, число замаскированных пикселей,
      число кандидатов, версия каталогов.
- [ ] Идемпотентность: повторный ingest того же `obsid` не создаёт дублей.
- [ ] Атомарная публикация JSON (temp + rename), health-check после обновления.
- [ ] Ручное подтверждение кандидатов до публикации «новых объектов».

---

## 10. Источники

Официальная документация и ноутбуки (проверено в ноябре 2026):

1. SPHEREx Explanatory Supplement (QR):
   `https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf`
2. SPHEREx data products:
   `https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products`
3. IRSA SPHEREx online help (HDU, FLAGS, bandpasses):
   `https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/sp.html`
4. IRSA tutorial: Intro:
   `https://caltech-ipac.github.io/irsa-tutorials/spherex-intro`
5. IRSA tutorial: Cutouts:
   `https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts`
6. IRSA tutorial: PSF:
   `https://caltech-ipac.github.io/irsa-tutorials/spherex-psf`
7. IRSA tutorial: Source Discovery Tool:
   `https://caltech-ipac.github.io/irsa-tutorials/spherex-source-discovery-tool-demo`
8. IRSA TAP documentation:
   `https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html`
9. IRSA TAP + Astropy:
   `https://irsa.ipac.caltech.edu/docs/program_interface/astropy_TAP.html`
10. Gaia archive TAP:
    `https://gea.esac.esa.int/tap-server/tap`
11. Gaia DR3 data model (`gaia_source`):
    `https://gea.esac.esa.int/archive/documentation/GDR3/`
12. CatWISE2020 в IRSA:
    `https://irsa.ipac.caltech.edu/Missions/catwise.html`
13. AllWISE в IRSA:
    `https://irsa.ipac.caltech.edu/Missions/wise.html`
14. SPHEREx 102-channel/17-channel описание (ApJS):
    `https://iopscience.iop.org/article/10.3847/1538-4365/ae522c`

Если таблица FLAGS или имена колонок в TAP разойдутся с этим документом —
правда в заголовке FITS и в `TAP_SCHEMA`, а не в этом файле. Читайте
`MSKNnnnn`/`MSKDnnnn` и проверяйте схему перед продакшеном.

