from datetime import date, datetime, timedelta
import hashlib
import hmac
import re
import secrets
import sqlite3
from threading import Lock

import pandas as pd


# Даты фиксированы, чтобы примеры и тесты не зависели от текущего дня.
FLIGHTS = [
    ("SU123", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-10T09:00", "вовремя", 2),
    ("SU123", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-12T09:00", "задержан", 1),
    ("SU124", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-12T18:00", "вовремя", 3),
    ("S7100", "S7", "Москва", "Казань", "2027-06-10T12:30", "вовремя", 0),
]

# Регулярные рейсы на разных маршрутах; исходные сценарии выше остаются отдельными.
_ROUTES = [
    ("SU201", "Аэрофлот", "Москва", "Сочи", 8, 15),
    ("SU202", "Аэрофлот", "Сочи", "Москва", 12, 40),
    ("S7201", "S7", "Москва", "Новосибирск", 7, 30),
    ("S7202", "S7", "Новосибирск", "Москва", 15, 20),
    ("DP301", "Победа", "Москва", "Калининград", 10, 10),
    ("DP302", "Победа", "Казань", "Санкт-Петербург", 13, 45),
    ("U6401", "Уральские авиалинии", "Екатеринбург", "Сочи", 6, 50),
    ("U6402", "Уральские авиалинии", "Санкт-Петербург", "Екатеринбург", 17, 5),
    ("UT501", "ЮТэйр", "Тюмень", "Москва", 9, 25),
    ("UT502", "ЮТэйр", "Москва", "Тюмень", 19, 30),
    ("N4601", "Nordwind", "Санкт-Петербург", "Казань", 11, 35),
    ("N4602", "Nordwind", "Сочи", "Санкт-Петербург", 16, 55),
]
_STATUSES = ("вовремя", "вовремя", "вовремя", "задержан", "отменён")
for route_index, (number, airline, origin, destination, hour, minute) in enumerate(_ROUTES):
    for day_offset in range(18):
        departure = datetime.combine(date(2027, 6, 1) + timedelta(days=day_offset), datetime.min.time())
        departure = departure.replace(hour=hour, minute=minute)
        status = _STATUSES[(route_index + day_offset) % len(_STATUSES)]
        FLIGHTS.append((
            number, airline, origin, destination, departure.isoformat(timespec="minutes"),
            status, 0 if status == "отменён" else (route_index * 13 + day_offset * 7) % 121,
        ))

COLUMNS = ["flight_number", "airline", "origin", "destination", "departure", "status", "seats"]
PASSPORT = re.compile(r"^[0-9]{10}$")
SEARCH_LIMIT = 10
CITY_FORMS = {
    "москва": ("москвы", "москву", "москве", "москвой"),
    "санкт-петербург": ("санкт-петербурга", "санкт-петербургу", "санкт-петербурге", "петербург", "петербурга", "петербурге", "питер", "питера", "питер"),
    "казань": ("казани", "казанью"),
    "сочи": (),
    "новосибирск": ("новосибирска", "новосибирску", "новосибирске", "новосибирском"),
    "калининград": ("калининграда", "калининграду", "калининграде", "калининградом"),
    "екатеринбург": ("екатеринбурга", "екатеринбургу", "екатеринбурге", "екатеринбургом"),
    "тюмень": ("тюмени", "тюменью"),
}


def normalize_city(value):
    value = value.strip().casefold().replace("ё", "е")
    return next((city for city, forms in CITY_FORMS.items() if value == city or value in forms), value)


def parse_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class FlightStore:
    def __init__(self, db_path=None):
        self._db = sqlite3.connect(str(db_path) if db_path is not None else ":memory:", timeout=30, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = Lock()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("CREATE TABLE IF NOT EXISTS flights (flight_number TEXT NOT NULL, airline TEXT NOT NULL, origin TEXT NOT NULL, destination TEXT NOT NULL, departure TEXT NOT NULL, status TEXT NOT NULL, seats INTEGER NOT NULL CHECK(seats >= 0))")
                self._db.execute("CREATE TABLE IF NOT EXISTS bookings (id INTEGER PRIMARY KEY AUTOINCREMENT, flight_number TEXT NOT NULL, flight_date TEXT NOT NULL, passport_fingerprint TEXT NOT NULL, UNIQUE(flight_number, flight_date, passport_fingerprint))")
                # Архив сохраняет все столбцы старой схемы, не перестраивая bookings
                # и не изменяя её существующие ограничения уникальности.
                self._db.execute("CREATE TABLE IF NOT EXISTS flight_store_cancelled_bookings AS SELECT * FROM bookings WHERE 0")
                for table in ("bookings", "flight_store_cancelled_bookings"):
                    columns = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
                    if "flight_rowid" not in columns:
                        self._db.execute(f"ALTER TABLE {table} ADD COLUMN flight_rowid INTEGER")
                if "passport" not in {row[1] for row in self._db.execute("PRAGMA table_info(bookings)")}:
                    self._db.execute("ALTER TABLE bookings ADD COLUMN passport TEXT")
                archived_columns = {row[1] for row in self._db.execute("PRAGMA table_info(flight_store_cancelled_bookings)")}
                for row in self._db.execute("PRAGMA table_info(bookings)").fetchall():
                    if row[1] not in archived_columns:
                        name = '"' + row[1].replace('"', '""') + '"'
                        self._db.execute(f"ALTER TABLE flight_store_cancelled_bookings ADD COLUMN {name}")
                self._db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
                self._db.execute("CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY CHECK(id = 1), name TEXT NOT NULL)")
                if "passport" not in {row[1] for row in self._db.execute("PRAGMA table_info(profile)")}:
                    self._db.execute("ALTER TABLE profile ADD COLUMN passport TEXT")
                if not self._db.execute("SELECT 1 FROM metadata WHERE key = 'initialized'").fetchone():
                    if not self._db.execute("SELECT 1 FROM flights LIMIT 1").fetchone():
                        self._db.executemany("INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?)", FLIGHTS)
                    self._db.execute("INSERT INTO metadata VALUES ('initialized', '1')")
                # Старые записи связываем только при единственном совпадении.
                if not self._db.execute("SELECT 1 FROM metadata WHERE key = 'booking_flight_migration'").fetchone():
                    self._db.execute("UPDATE bookings SET flight_rowid = (SELECT MIN(rowid) FROM flights f WHERE f.flight_number = bookings.flight_number AND substr(f.departure, 1, 10) = bookings.flight_date) WHERE flight_rowid IS NULL AND (SELECT COUNT(*) FROM flights f WHERE f.flight_number = bookings.flight_number AND substr(f.departure, 1, 10) = bookings.flight_date) = 1")
                    self._db.execute("INSERT INTO metadata VALUES ('booking_flight_migration', '1')")
                self._db.execute("INSERT OR IGNORE INTO metadata VALUES ('passport_key', ?)", (secrets.token_bytes(32),))
                self._passport_key = self._db.execute("SELECT value FROM metadata WHERE key = 'passport_key'").fetchone()[0]
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @property
    def flights(self):
        with self._lock:
            return self._flights()

    def _flights(self):
        return pd.DataFrame.from_records([tuple(row) for row in self._db.execute("SELECT flight_number, airline, origin, destination, departure, status, seats FROM flights ORDER BY rowid")], columns=COLUMNS)

    @property
    def bookings(self):
        with self._lock:
            return [{"id": f"DEMO-{row['id']:04d}", "flight_number": row["flight_number"], "flight_date": row["flight_date"], "passport_fingerprint": row["passport_fingerprint"]} for row in self._db.execute("SELECT * FROM bookings ORDER BY id")]

    def close(self):
        with self._lock:
            self._db.close()

    def get_profile(self):
        with self._lock:
            row = self._db.execute("SELECT name FROM profile WHERE id = 1").fetchone()
        return {"status": "ok", "name": row[0]} if row else {"status": "not_found", "message": "Имя пользователя не сохранено."}

    def set_profile(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[^\W\d_]+(?:[ '\u2019-][^\W\d_]+)*", name.strip(), re.UNICODE) or len(name.strip()) > 200:
            return {"status": "error", "message": "Укажите корректное имя пользователя."}
        name = name.strip()
        with self._lock:
            self._db.execute("INSERT INTO profile (id, name) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET passport = CASE WHEN name = excluded.name THEN passport ELSE NULL END, name = excluded.name", (name,))
        return {"status": "ok", "name": name}

    def get_profile_passport(self, expected_name: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT passport FROM profile WHERE id = 1 AND name = ?", (expected_name,)).fetchone()
        return row[0] or "" if row else ""

    def set_profile_passport(self, passport: str, expected_name: str) -> bool:
        if not isinstance(passport, str) or not PASSPORT.fullmatch(passport):
            return False
        with self._lock:
            return self._db.execute("UPDATE profile SET passport = ? WHERE id = 1 AND name = ?", (passport, expected_name)).rowcount == 1

    def delete_profile(self):
        with self._lock:
            self._db.execute("DELETE FROM profile WHERE id = 1")
        return {"status": "ok", "message": "Профиль удалён."}

    def search(self, origin: str = "", destination: str = "", flight_date: str = "", available_only: bool = False, limit: int = SEARCH_LIMIT, from_now: bool = False) -> dict:
        """Поиск по фиксированному учебному расписанию; count не ограничен limit."""
        if flight_date and not parse_date(flight_date):
            return {"status": "error", "message": "Дата должна быть в формате YYYY-MM-DD."}
        limit = max(1, min(limit, SEARCH_LIMIT))
        with self._lock:
            matches = self._flights()
            for column, value in (("origin", origin), ("destination", destination)):
                if value.strip():
                    matches = matches[matches[column].map(normalize_city) == normalize_city(value)]
            if flight_date:
                matches = matches[matches.departure.str[:10] == flight_date]
            if from_now:
                matches = matches[pd.to_datetime(matches.departure) >= datetime.now()]
            if available_only:
                matches = matches[(matches.seats > 0) & (matches.status != "отменён")]
            matches = matches.sort_values(["departure", "flight_number", "origin", "destination"], kind="stable")
            count = len(matches)
            flights = matches.head(limit).to_dict("records")
        return {"status": "ok", "message": "Поиск выполнен.",
                "count": count, "flights": flights, "truncated": count > len(flights)}

    def status(self, flight_number: str, flight_date: str = "") -> dict:
        if not flight_number or not flight_number.strip():
            return {"status": "missing_fields", "message": "Укажите номер рейса.", "missing_fields": ["flight_number"]}
        flights = self.flights
        matches = flights[flights.flight_number == flight_number.strip().upper()]
        if flight_date:
            if not parse_date(flight_date):
                return {"status": "error", "message": "Дата должна быть в формате YYYY-MM-DD."}
            matches = matches[matches.departure.str[:10] == flight_date]
        if matches.empty:
            return {"status": "not_found", "message": "Рейс не найден."}
        if len(matches) > 1:
            if flight_date:
                return {"status": "error", "message": "Найдено несколько рейсов с этим номером и датой."}
            return {"status": "missing_fields", "message": "Укажите дату рейса.", "missing_fields": ["flight_date"]}
        return {"status": "ok", "flight": matches.iloc[0].to_dict()}

    def book(self, flight_number: str, flight_date: str, passenger_name: str, passport: str, airline: str = "", origin: str = "", destination: str = "", *, remember_passport: bool = False) -> dict:
        required = {"flight_number": flight_number, "flight_date": flight_date, "passenger_name": passenger_name, "passport": passport}
        missing = [key for key, value in required.items() if not isinstance(value, str) or not value.strip()]
        if missing:
            return {"status": "missing_fields", "message": "Укажите недостающие данные.", "missing_fields": missing}
        if not PASSPORT.fullmatch(passport):
            return {"status": "error", "message": "Паспорт должен состоять из 10 цифр."}
        requested = parse_date(flight_date)
        if not requested:
            return {"status": "error", "message": "Дата должна быть в формате YYYY-MM-DD."}
        number = flight_number.strip().upper()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                result = self._book(number, flight_date, passport, airline, origin, destination, requested)
                if remember_passport and result["status"] in ("booked", "duplicate"):
                    self._db.execute("UPDATE profile SET passport = ? WHERE id = 1 AND name = ?", (passport, passenger_name.strip()))
                self._db.commit()
                return result
            except BaseException:
                self._db.rollback()
                raise

    def _book(self, number, flight_date, passport, airline, origin, destination, requested):
            flights = self._flights()
            same_number = flights[flights.flight_number == number]
            if same_number.empty:
                return {"status": "not_found", "message": "Рейс не найден; проверьте авиакомпанию и маршрут."}
            for column, requested_value in (("airline", airline), ("origin", origin), ("destination", destination)):
                if requested_value:
                    if column in ("origin", "destination"):
                        same_number = same_number[same_number[column].map(normalize_city) == normalize_city(requested_value)]
                    else:
                        same_number = same_number[same_number[column].str.casefold() == requested_value.strip().casefold()]
            if same_number.empty:
                return {"status": "not_found", "message": "Указанная авиакомпания или маршрут не соответствует рейсу."}
            matches = same_number[same_number.departure.str[:10] == flight_date]
            if matches.empty:
                flight = same_number.iloc[0]
                route = flights[
                    (flights.airline == flight.airline)
                    & (flights.origin == flight.origin)
                    & (flights.destination == flight.destination)
                    & (flights.seats > 0)
                    & (flights.status != "отменён")
                ]
                # Сначала расстояние до даты, затем будущий рейс, затем время вылета.
                options = sorted(route.to_dict("records"), key=lambda f: (
                    abs((datetime.fromisoformat(f["departure"]).date() - requested).days),
                    datetime.fromisoformat(f["departure"]).date() < requested,
                    f["departure"],
                ))
                return {"status": "not_found", "message": "На эту дату рейса нет.", "alternatives": options}
            if len(matches) > 1:
                return {"status": "error", "message": "Найдено несколько рейсов; уточните авиакомпанию и маршрут."}
            index = matches.index[0]
            fingerprint = hmac.new(self._passport_key, passport.encode(), hashlib.sha256).hexdigest()
            duplicate = self._db.execute("SELECT id FROM bookings WHERE flight_number = ? AND flight_date = ? AND passport_fingerprint = ?", (number, flight_date, fingerprint)).fetchone()
            if duplicate:
                self._db.execute("UPDATE bookings SET passport = ? WHERE id = ? AND passport IS NULL", (passport, duplicate["id"]))
                return {"status": "duplicate", "message": "Бронь для этого пассажира уже существует.", "booking_id": f"DEMO-{duplicate['id']:04d}", "flight_number": number, "flight_date": flight_date}
            if flights.at[index, "seats"] <= 0 or flights.at[index, "status"] == "отменён":
                return {"status": "sold_out", "message": "Свободных мест нет."}
            rowid = self._db.execute("SELECT rowid FROM flights ORDER BY rowid LIMIT 1 OFFSET ?", (int(index),)).fetchone()[0]
            self._db.execute("UPDATE flights SET seats = seats - 1 WHERE rowid = ?", (rowid,))
            # Не переиспользовать ID отменённой брони даже в старой схеме без AUTOINCREMENT.
            next_id = self._db.execute("SELECT MAX(id) FROM (SELECT id FROM bookings UNION ALL SELECT id FROM flight_store_cancelled_bookings)").fetchone()[0]
            if self._db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'").fetchone():
                sequence = self._db.execute("SELECT seq FROM sqlite_sequence WHERE name = 'bookings'").fetchone()
                next_id = max(next_id or 0, sequence[0] if sequence else 0)
            cursor = self._db.execute('INSERT INTO bookings (id, flight_number, flight_date, passport_fingerprint, flight_rowid, passport) VALUES (?, ?, ?, ?, ?, ?)', ((next_id or 0) + 1, number, flight_date, fingerprint, rowid, passport))
            booking_id = f"DEMO-{cursor.lastrowid:04d}"
            return {"status": "booked", "message": "Бронь создана.", "booking_id": booking_id, "flight_number": number, "flight_date": flight_date}

    def find_booking(self, flight_number, flight_date, passport):
        """Найти существующую бронь после прерывания вызова без повторной записи."""
        if not isinstance(flight_number, str) or not parse_date(flight_date) or not isinstance(passport, str) or not PASSPORT.fullmatch(passport):
            return None
        number = flight_number.strip().upper()
        fingerprint = hmac.new(self._passport_key, passport.encode(), hashlib.sha256).hexdigest()
        with self._lock:
            booking = self._db.execute("SELECT id FROM bookings WHERE flight_number = ? AND flight_date = ? AND passport_fingerprint = ?", (number, flight_date, fingerprint)).fetchone()
            if booking:
                return {"status": "duplicate", "message": "Бронь для этого пассажира уже существует.", "booking_id": f"DEMO-{booking['id']:04d}", "flight_number": number, "flight_date": flight_date}
        return None

    @staticmethod
    def _booking_id(value):
        if not isinstance(value, str) or not re.fullmatch(r"DEMO-[0-9]{4,19}", value):
            return None
        number = int(value[5:])
        return number if 0 < number <= 9223372036854775807 and value == f"DEMO-{number:04d}" else None

    def _booking_flight(self, number, flight_date, flight_rowid=None, allow_unbound=False):
        rows = self._db.execute(
            "SELECT rowid, airline, origin, destination, departure, status, seats "
            "FROM flights WHERE flight_number = ? AND substr(departure, 1, 10) = ? ORDER BY rowid",
            (number, flight_date),
        ).fetchall()
        if flight_rowid is not None:
            return next((row for row in rows if row["rowid"] == flight_rowid), None)
        return rows[0] if allow_unbound and len(rows) == 1 else None

    def list_bookings(self) -> dict:
        """Активные брони общего локального пользователя, без паспортных данных.

        Возвращает {status: 'ok', bookings: [...]}; у каждой записи booking_id,
        flight_number, flight_date, airline, origin, destination, departure,
        flight_status. Если рейс отсутствует в расписании, его поля равны None.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT b.id, b.flight_number, b.flight_date, f.airline, f.origin, "
                "f.destination, f.departure, f.status AS flight_status FROM bookings b "
                "LEFT JOIN flights f ON f.rowid = b.flight_rowid AND f.flight_number = b.flight_number "
                "AND substr(f.departure, 1, 10) = b.flight_date ORDER BY b.id"
            ).fetchall()
        return {"status": "ok", "bookings": [
            {"booking_id": f"DEMO-{row['id']:04d}", **{key: row[key] for key in (
                "flight_number", "flight_date", "airline", "origin", "destination", "departure", "flight_status"
            )}} for row in rows
        ]}

    def cancel_booking(self, booking_id: str) -> dict:
        """Отменить бронь DEMO-NNNN и вернуть место ровно один раз.

        Статусы: cancelled (включая повторную отмену), not_found, error.
        Повторная отмена возвращает already_cancelled=True; неизвестный рейс
        даёт error без изменений. Ошибки SQLite пробрасываются после rollback.
        """
        ident = self._booking_id(booking_id)
        if ident is None:
            return {"status": "error", "message": "Некорректный номер брони."}
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                booking = self._db.execute("SELECT * FROM bookings WHERE id = ?", (ident,)).fetchone()
                if not booking:
                    archived = self._db.execute("SELECT 1 FROM flight_store_cancelled_bookings WHERE id = ? LIMIT 1", (ident,)).fetchone()
                    result = ({"status": "cancelled", "booking_id": booking_id, "already_cancelled": True}
                              if archived else {"status": "not_found", "message": "Бронь не найдена."})
                else:
                    flight = self._booking_flight(booking["flight_number"], booking["flight_date"], booking["flight_rowid"])
                    if not flight:
                        result = {"status": "error", "message": "Рейс брони отсутствует в расписании или определить его однозначно невозможно."}
                    else:
                        source = {row[1] for row in self._db.execute("PRAGMA table_info(bookings)")}
                        target = [row[1] for row in self._db.execute("PRAGMA table_info(flight_store_cancelled_bookings)")]
                        columns = [name for name in target if name in source and name != "passport"]
                        quoted = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
                        self._db.execute(f"INSERT INTO flight_store_cancelled_bookings ({quoted}) SELECT {quoted} FROM bookings WHERE id = ?", (ident,))
                        self._db.execute("DELETE FROM bookings WHERE id = ?", (ident,))
                        self._db.execute("UPDATE flights SET seats = seats + 1 WHERE rowid = ?", (flight["rowid"],))
                        result = {"status": "cancelled", "booking_id": booking_id, "already_cancelled": False}
                self._db.commit()
                return result
            except BaseException:
                self._db.rollback()
                raise

    def update_booking(self, booking_id: str, flight_number: str, flight_date: str) -> dict:
        """Перенести активную бронь на указанный рейс и дату YYYY-MM-DD.

        Статусы: updated, not_found (бронь или новый рейс), sold_out,
        duplicate (тот же пассажир уже забронировал рейс), error. Перенос на
        тот же рейс возвращает updated с unchanged=True. При ошибке бронь и
        места не меняются; ошибки SQLite пробрасываются после rollback.
        """
        ident = self._booking_id(booking_id)
        if ident is None or not isinstance(flight_number, str) or not flight_number.strip() or not parse_date(flight_date):
            return {"status": "error", "message": "Укажите корректные номер брони, рейс и дату YYYY-MM-DD."}
        number = flight_number.strip().upper()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                booking = self._db.execute("SELECT * FROM bookings WHERE id = ?", (ident,)).fetchone()
                if not booking:
                    result = {"status": "not_found", "message": "Активная бронь не найдена."}
                else:
                    old = self._booking_flight(booking["flight_number"], booking["flight_date"], booking["flight_rowid"])
                    new = self._booking_flight(number, flight_date, allow_unbound=True)
                    duplicate = self._db.execute(
                        "SELECT id FROM bookings WHERE flight_number = ? AND flight_date = ? AND passport_fingerprint = ? AND id != ?",
                        (number, flight_date, booking["passport_fingerprint"], ident),
                    ).fetchone()
                    if not old:
                        result = {"status": "error", "message": "Исходный рейс отсутствует в расписании или определить его однозначно невозможно."}
                    elif booking["flight_number"] == number and booking["flight_date"] == flight_date:
                        result = {"status": "updated", "booking_id": booking_id, "flight_number": number, "flight_date": flight_date, "unchanged": True}
                    elif not new:
                        count = self._db.execute("SELECT COUNT(*) FROM flights WHERE flight_number = ? AND substr(departure, 1, 10) = ?", (number, flight_date)).fetchone()[0]
                        result = {"status": "error" if count else "not_found", "message": "Новый рейс неоднозначен; уточните рейс." if count else "Новый рейс не найден."}
                    elif duplicate:
                        result = {"status": "duplicate", "message": "Бронь для этого пассажира уже существует.", "booking_id": f"DEMO-{duplicate['id']:04d}"}
                    elif new["seats"] <= 0 or new["status"] == "отменён":
                        result = {"status": "sold_out", "message": "Свободных мест нет."}
                    else:
                        self._db.execute("UPDATE bookings SET flight_number = ?, flight_date = ?, flight_rowid = ? WHERE id = ?", (number, flight_date, new["rowid"], ident))
                        self._db.execute("UPDATE flights SET seats = seats - 1 WHERE rowid = ?", (new["rowid"],))
                        self._db.execute("UPDATE flights SET seats = seats + 1 WHERE rowid = ?", (old["rowid"],))
                        result = {"status": "updated", "booking_id": booking_id, "flight_number": number, "flight_date": flight_date, "unchanged": False}
                self._db.commit()
                return result
            except BaseException:
                self._db.rollback()
                raise
