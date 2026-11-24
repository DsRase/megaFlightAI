from datetime import date, datetime

from flight_agent.data import COLUMNS, FLIGHTS, FlightStore


def test_flight_dataset_quality_and_determinism():
    first = FlightStore().flights
    second = FlightStore().flights

    assert first.equals(second)
    assert list(first.columns) == COLUMNS
    assert len(first) >= 200
    assert len(first[["flight_number", "departure"]].assign(
        departure=first.departure.str[:10]
    ).drop_duplicates()) == len(first)
    assert first.notna().all().all()
    assert all(isinstance(value, str) and value.strip()
               for column in COLUMNS[:-1] for value in first[column])
    assert all(row.origin != row.destination for row in first.itertuples())
    assert all(date(2027, 6, 1) <= datetime.fromisoformat(value).date() <= date(2027, 6, 18)
               for value in first.departure)
    assert all(isinstance(value, int) and 0 <= value <= 120 for value in first.seats)
    assert {"вовремя", "задержан", "отменён"} <= set(first.status)
    assert (first.loc[first.status == "отменён", "seats"] == 0).all()
    assert len(set(first.airline)) >= 6
    assert len(set(zip(first.origin, first.destination))) >= 12
    assert first.departure.str[:10].nunique() >= 18
    assert 0 in set(first.seats) and first.seats.max() > 50
    assert FLIGHTS[:4] == [
        ("SU123", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-10T09:00", "вовремя", 2),
        ("SU123", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-12T09:00", "задержан", 1),
        ("SU124", "Аэрофлот", "Москва", "Санкт-Петербург", "2027-06-12T18:00", "вовремя", 3),
        ("S7100", "S7", "Москва", "Казань", "2027-06-10T12:30", "вовремя", 0),
    ]


def test_generated_flight_booking_and_route_alternatives():
    store = FlightStore()
    assert store.status("DP301", "2027-06-02")["flight"]["destination"] == "Калининград"
    assert store.book("DP301", "2027-06-02", "Иван", "1234567890")["status"] == "booked"
    alternatives = store.book("DP301", "2027-06-19", "Иван", "1234567890")["alternatives"]
    assert len(alternatives) < 18
    assert alternatives[0]["departure"] == "2027-06-18T10:10"
    assert all(f["airline"] == "Победа" and f["origin"] == "Москва"
               and f["destination"] == "Калининград" for f in alternatives)
    assert all(f["seats"] > 0 and f["status"] != "отменён" for f in alternatives)


def test_booking_alternatives_exclude_cancelled_and_exhausted_flights():
    store = FlightStore()
    store._db.execute("UPDATE flights SET status = 'отменён' WHERE flight_number = 'SU123' AND departure LIKE '2027-06-10%'")
    store._db.execute("UPDATE flights SET seats = 0 WHERE flight_number = 'SU123' AND departure LIKE '2027-06-12%'")
    result = store.book("SU123", "2027-06-11", "Иван", "1234567890")
    assert [flight["flight_number"] for flight in result["alternatives"]] == ["SU124"]
    store._db.execute("UPDATE flights SET seats = 0 WHERE flight_number = 'SU124'")
    assert store.book("SU123", "2027-06-11", "Иван", "1234567890")["alternatives"] == []
    store._db.execute("UPDATE flights SET seats = 5 WHERE flight_number = 'SU123' AND status = 'отменён'")
    assert store.book("SU123", "2027-06-10", "Иван", "1234567890")["status"] == "sold_out"


def test_status_found_delayed_missing_and_unknown():
    store = FlightStore()
    assert store.status("SU123", "2027-06-10")["flight"]["status"] == "вовремя"
    assert store.status("SU123", "2027-06-12")["flight"]["status"] == "задержан"
    assert store.status("SU123")["missing_fields"] == ["flight_date"]
    assert store.status("")["missing_fields"] == ["flight_number"]
    assert store.status("XX000")["status"] == "not_found"


def test_booking_validation_capacity_and_duplicates():
    store = FlightStore()
    assert store.book("SU123", "2027-06-10", "Иван", "123")["status"] == "error"
    assert store.book("SU123", "2027-06-10", "Иван", "")["missing_fields"] == ["passport"]
    assert store.book("S7100", "2027-06-10", "Иван", "1234567890")["status"] == "sold_out"
    assert store.book("XX000", "2027-06-10", "Иван", "1234567890")["status"] == "not_found"
    assert store.book("SU123", "2027-06-10", "Иван", "1234567890", airline="S7", origin="Москва", destination="Казань")["status"] == "not_found"
    booking = store.book("SU123", "2027-06-10", "Иван", "1234567890")
    duplicate = store.book("SU123", "2027-06-10", "Иван", "1234567890")
    assert booking["status"] == "booked"
    assert duplicate["status"] == "duplicate"
    assert duplicate["booking_id"] == booking["booking_id"]
    assert "1234567890" not in str(store.bookings)
    assert len(store.bookings) == 1


def test_alternatives_are_nearest_future_first():
    store = FlightStore()
    result = store.book("SU123", "2027-06-11", "Иван", "1234567890")
    assert result["status"] == "not_found"
    assert [f["departure"] for f in result["alternatives"]] == ["2027-06-12T09:00", "2027-06-12T18:00", "2027-06-10T09:00"]
    assert not store.bookings


def test_booking_uses_same_city_forms_as_search():
    store = FlightStore()
    assert store.search(origin="москвы", destination="питера", flight_date="2027-06-12")["count"] > 0
    assert store.book("SU123", "2027-06-12", "Иван", "1234567890", origin="москвы", destination="питера")["status"] == "booked"
    assert store.book("SU124", "2027-06-12", "Иван", "1234567890", destination="Казани")["status"] == "not_found"


def test_date_format_is_consistent_across_tools():
    store = FlightStore()
    for value in ("20270610", "2027-W23-4", "2027-02-30", "2027-6-10"):
        assert store.search(flight_date=value)["status"] == "error"
        assert store.status("SU123", value)["status"] == "error"
        assert store.book("SU123", value, "Иван", "1234567890")["status"] == "error"
