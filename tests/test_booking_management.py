from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

import pytest

from flight_agent.data import FlightStore


def seats(store, number, day):
    return store.status(number, day)["flight"]["seats"]


def test_list_cancel_and_rebook_without_exposing_passport(tmp_path):
    path = tmp_path / "flights.db"
    store = FlightStore(path)
    booking = store.book("SU123", "2027-06-10", "Иван", "1234567890")
    ident = booking["booking_id"]
    entry = store.list_bookings()["bookings"][0]
    assert entry == {"booking_id": ident, "flight_number": "SU123", "flight_date": "2027-06-10",
                     "airline": "Аэрофлот", "origin": "Москва", "destination": "Санкт-Петербург",
                     "departure": "2027-06-10T09:00", "flight_status": "вовремя"}
    assert "1234567890" not in str(entry) and "fingerprint" not in str(entry)
    assert store.cancel_booking(ident)["already_cancelled"] is False
    assert store.cancel_booking(ident)["already_cancelled"] is True
    assert seats(store, "SU123", "2027-06-10") == 2
    assert store.list_bookings()["bookings"] == []
    replacement = store.book("SU123", "2027-06-10", "Иван", "1234567890")
    assert replacement["status"] == "booked" and replacement["booking_id"] != ident
    store.close()
    store = FlightStore(path)
    assert store.cancel_booking(ident)["already_cancelled"] is True
    assert len(store.list_bookings()["bookings"]) == 1
    assert seats(store, "SU123", "2027-06-10") == 1
    store.close()


def test_transfer_failures_preserve_booking_and_seats():
    store = FlightStore()
    ident = store.book("SU123", "2027-06-10", "Иван", "1234567890")["booking_id"]
    other = store.book("SU124", "2027-06-12", "Иван", "1234567890")["booking_id"]
    original = store.list_bookings()
    for number, day, expected in [("XX000", "2027-06-12", "not_found"),
                                  ("S7100", "2027-06-10", "sold_out"),
                                  ("SU124", "2027-06-12", "duplicate")]:
        assert store.update_booking(ident, number, day)["status"] == expected
        assert store.list_bookings() == original
        assert seats(store, "SU123", "2027-06-10") == 1
    assert store.update_booking(ident, "SU123", "2027-06-10")["unchanged"] is True
    assert store.update_booking(ident, "su123", "2027-06-12")["status"] == "updated"
    assert seats(store, "SU123", "2027-06-10") == 2
    assert seats(store, "SU123", "2027-06-12") == 0
    assert {entry["booking_id"] for entry in store.list_bookings()["bookings"]} == {ident, other}
    store.close()


def test_transaction_rolls_back_on_database_error():
    store = FlightStore()
    ident = store.book("SU123", "2027-06-10", "Иван", "1234567890")["booking_id"]
    store._db.execute("CREATE TRIGGER fail_seats BEFORE UPDATE OF seats ON flights BEGIN SELECT RAISE(ABORT, 'failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.update_booking(ident, "SU124", "2027-06-12")
    with pytest.raises(sqlite3.IntegrityError):
        store.cancel_booking(ident)
    assert store.list_bookings()["bookings"][0]["flight_number"] == "SU123"
    assert seats(store, "SU123", "2027-06-10") == 1
    assert seats(store, "SU124", "2027-06-12") == 3
    store.close()


def test_concurrent_transfers_and_cancellation(tmp_path):
    path = tmp_path / "shared.db"
    stores = [FlightStore(path) for _ in range(2)]
    ids = [stores[0].book("SU123", "2027-06-10", "Иван", passport)["booking_id"]
           for passport in ("1234567890", "9876543210")]
    barrier = Barrier(2)

    def transfer(index):
        barrier.wait()
        return stores[index].update_booking(ids[index], "SU123", "2027-06-12")["status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(transfer, range(2))) == ["sold_out", "updated"]
    barrier = Barrier(2)

    def cancel(index):
        barrier.wait()
        return stores[index].cancel_booking(ids[0])["already_cancelled"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(cancel, range(2))) == [False, True]
    assert sum(seats(stores[0], "SU123", day) for day in ("2027-06-10", "2027-06-12")) == 2
    for store in stores:
        store.close()


def test_legacy_schema_extra_columns_and_ids_are_preserved(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE bookings (id INTEGER PRIMARY KEY, flight_number TEXT NOT NULL, flight_date TEXT NOT NULL, passport_fingerprint TEXT NOT NULL, note TEXT DEFAULT 'legacy', UNIQUE(flight_number, flight_date, passport_fingerprint))")
        db.execute("INSERT INTO bookings VALUES (42, 'SU123', '2027-06-10', 'old-fingerprint', 'keep me')")
    store = FlightStore(path)
    assert store.list_bookings()["bookings"][0]["booking_id"] == "DEMO-0042"
    assert store.cancel_booking("DEMO-0042")["status"] == "cancelled"
    assert store._db.execute("SELECT note FROM flight_store_cancelled_bookings WHERE id = 42").fetchone()[0] == "keep me"
    assert store.book("SU123", "2027-06-10", "Иван", "1234567890")["booking_id"] == "DEMO-0043"
    store.close()


def test_missing_flight_and_invalid_identifiers():
    store = FlightStore()
    ident = store.book("SU123", "2027-06-10", "Иван", "1234567890")["booking_id"]
    store._db.execute("DELETE FROM flights WHERE flight_number = 'SU123' AND departure LIKE '2027-06-10%'")
    assert store.list_bookings()["bookings"][0]["departure"] is None
    assert store.cancel_booking(ident)["status"] == "error"
    assert store.update_booking(ident, "SU124", "2027-06-12")["status"] == "error"
    assert store.cancel_booking("DEMO-9999")["status"] == "not_found"
    assert store.cancel_booking("DEMO-00001")["status"] == "error"
    assert store.update_booking(ident, "SU124", "2027-02-30")["status"] == "error"
    assert len(store.list_bookings()["bookings"]) == 1
    store.close()


def test_booking_tracks_selected_flight_and_legacy_ambiguity(tmp_path):
    path = tmp_path / "ambiguous.db"
    store = FlightStore(path)
    store._db.execute("INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?)",
                      ("SU123", "Другая", "Казань", "Сочи", "2027-06-10T13:00", "вовремя", 4))
    assert store.book("SU123", "2027-06-10", "Иван", "1234567890")["status"] == "error"
    booking = store.book("SU123", "2027-06-10", "Иван", "1234567890", airline="Другая")
    assert store.list_bookings()["bookings"][0]["airline"] == "Другая"
    assert store.cancel_booking(booking["booking_id"])["status"] == "cancelled"
    assert store._db.execute("SELECT seats FROM flights WHERE airline = 'Другая'").fetchone()[0] == 4
    store._db.execute("INSERT INTO bookings (flight_number, flight_date, passport_fingerprint) VALUES ('SU123', '2027-06-10', 'legacy')")
    ident = f"DEMO-{store._db.execute('SELECT max(id) FROM bookings').fetchone()[0]:04d}"
    before = list(store._db.execute("SELECT rowid, seats FROM flights WHERE flight_number = 'SU123' AND departure LIKE '2027-06-10%'"))
    assert store.cancel_booking(ident)["status"] == "error"
    assert store.update_booking(ident, "SU124", "2027-06-12")["status"] == "error"
    assert list(store._db.execute("SELECT rowid, seats FROM flights WHERE flight_number = 'SU123' AND departure LIKE '2027-06-10%'")) == before
    store.close()
