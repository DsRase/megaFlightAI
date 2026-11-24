from datetime import date, datetime
import hashlib
import hmac
import re
import secrets
from threading import Lock

import pandas as pd


# Даты фиксированы, чтобы примеры и тесты не зависели от текущего дня.
FLIGHTS = [
    ("SU123", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-10T09:00", "вовремя", 2),
    ("SU123", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-12T09:00", "задержан", 1),
    ("SU124", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-12T18:00", "вовремя", 3),
    ("S7100", "S7", "Москва", "Казань", "2027-06-10T12:30", "вовремя", 0),
]
COLUMNS = ["flight_number", "airline", "origin", "destination", "departure", "status", "seats"]
PASSPORT = re.compile(r"^[0-9]{10}$")


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class FlightStore:
    def __init__(self):
        self.flights = pd.DataFrame(FLIGHTS, columns=COLUMNS)
        self.bookings = []
        self._passport_key = secrets.token_bytes(32)
        self._lock = Lock()

    def status(self, flight_number: str, flight_date: str = "") -> dict:
        if not flight_number or not flight_number.strip():
            return {"status": "missing_fields", "message": "Укажите номер рейса.", "missing_fields": ["flight_number"]}
        matches = self.flights[self.flights.flight_number == flight_number.strip().upper()]
        if flight_date:
            if not parse_date(flight_date):
                return {"status": "error", "message": "Дата должна быть в формате YYYY-MM-DD."}
            matches = matches[matches.departure.str[:10] == flight_date]
        if matches.empty:
            return {"status": "not_found", "message": "Рейс не найден."}
        if len(matches) > 1:
            return {"status": "missing_fields", "message": "Укажите дату рейса.", "missing_fields": ["flight_date"]}
        return {"status": "ok", "flight": matches.iloc[0].to_dict()}

    def book(self, flight_number: str, flight_date: str, passenger_name: str, passport: str, airline: str = "", origin: str = "", destination: str = "") -> dict:
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
            same_number = self.flights[self.flights.flight_number == number]
            if same_number.empty:
                return {"status": "not_found", "message": "Рейс не найден; проверьте авиакомпанию и маршрут."}
            for column, requested_value in (("airline", airline), ("origin", origin), ("destination", destination)):
                if requested_value:
                    same_number = same_number[same_number[column].str.casefold() == requested_value.strip().casefold()]
            if same_number.empty:
                return {"status": "not_found", "message": "Указанная авиакомпания или маршрут не соответствует рейсу."}
            matches = same_number[same_number.departure.str[:10] == flight_date]
            if matches.empty:
                flight = same_number.iloc[0]
                route = self.flights[
                    (self.flights.airline == flight.airline)
                    & (self.flights.origin == flight.origin)
                    & (self.flights.destination == flight.destination)
                ]
                # Сначала расстояние до даты, затем будущий рейс, затем время вылета.
                options = sorted(route.to_dict("records"), key=lambda f: (
                    abs((datetime.fromisoformat(f["departure"]).date() - requested).days),
                    datetime.fromisoformat(f["departure"]).date() < requested,
                    f["departure"],
                ))
                return {"status": "not_found", "message": "На эту дату рейса нет.", "alternatives": options}
            index = matches.index[0]
            fingerprint = hmac.new(self._passport_key, passport.encode(), hashlib.sha256).hexdigest()
            duplicate = next((b for b in self.bookings if b["flight_number"] == number and b["flight_date"] == flight_date and b["passport_fingerprint"] == fingerprint), None)
            if duplicate:
                return {"status": "duplicate", "message": "Бронь для этого пассажира уже существует.", "booking_id": duplicate["id"], "flight_number": number, "flight_date": flight_date}
            if self.flights.at[index, "seats"] <= 0:
                return {"status": "sold_out", "message": "Свободных мест нет."}
            booking_id = f"DEMO-{len(self.bookings) + 1:04d}"
            self.bookings.append({"id": booking_id, "flight_number": number, "flight_date": flight_date, "passport_fingerprint": fingerprint})
            self.flights.at[index, "seats"] -= 1
            return {"status": "booked", "message": "Демобронь создана.", "booking_id": booking_id, "flight_number": number, "flight_date": flight_date}
