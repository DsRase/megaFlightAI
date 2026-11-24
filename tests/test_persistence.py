from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

from flight_agent.data import FlightStore


def test_restart_preserves_booking_seats_profile_and_fingerprint(tmp_path):
    path = tmp_path / "flights.sqlite3"
    store = FlightStore(path)
    assert store.set_profile("Иван")["status"] == "ok"
    booking = store.book("SU123", "2027-06-12", "Иван", "0123456789", remember_passport=True)
    store.close()
    store = FlightStore(path)
    assert store.status("SU123", "2027-06-12")["flight"]["seats"] == 0
    assert store.get_profile()["name"] == "Иван"
    assert store.get_profile_passport(expected_name="Иван") == "0123456789"
    duplicate = store.book("SU123", "2027-06-12", "Иван", "0123456789")
    assert duplicate["status"] == "duplicate" and duplicate["booking_id"] == booking["booking_id"]
    assert store.find_booking("SU123", "2027-06-12", "0123456789")["booking_id"] == booking["booking_id"]
    assert store.book("SU123", "2027-06-12", "Анна", "9876543210")["status"] == "sold_out"
    store.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT passport FROM bookings").fetchone()[0] == "0123456789"
    assert "0123456789" not in str(booking) + str(duplicate)


def test_profile_update_delete_and_in_memory_isolation(tmp_path):
    path = tmp_path / "flights.sqlite3"
    store = FlightStore(path)
    assert store.get_profile()["status"] == "not_found"
    assert not store.set_profile_passport("0123456789", expected_name="Иван")
    assert store.set_profile("Иван")["name"] == "Иван"
    assert not store.set_profile_passport("123", expected_name="Иван")
    assert store.set_profile_passport("0123456789", expected_name="Иван")
    assert store.set_profile("Иван")["name"] == "Иван"
    assert store.get_profile_passport(expected_name="Иван") == "0123456789"
    assert store.set_profile("Анна")["name"] == "Анна"
    assert store.get_profile_passport(expected_name="Иван") == ""
    assert not store.set_profile_passport("9876543210", expected_name="Иван")
    assert store.get_profile_passport(expected_name="Анна") == ""
    assert store.set_profile("1234567890")["status"] == "error"
    store.close()
    store = FlightStore(path)
    assert store.get_profile()["name"] == "Анна"
    store.delete_profile()
    store.close()
    reopened = FlightStore(path)
    assert reopened.get_profile()["status"] == "not_found"
    assert reopened.get_profile_passport(expected_name="Анна") == ""
    assert FlightStore().get_profile()["status"] == "not_found"
    assert FlightStore().status("SU123", "2027-06-12")["flight"]["seats"] == 1
    reopened.close()


def test_legacy_migration_duplicate_transfer_and_cancel(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    store = FlightStore(path)
    store.set_profile("Иван")
    booking = store.book("SU123", "2027-06-10", "Иван", "0123456789")
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE bookings DROP COLUMN passport")
        db.execute("ALTER TABLE profile DROP COLUMN passport")
    store = FlightStore(path)
    assert store.get_profile_passport(expected_name="Иван") == ""
    assert store.book("SU123", "2027-06-10", "Иван", "0123456789", remember_passport=True)["status"] == "duplicate"
    assert store.get_profile_passport(expected_name="Иван") == "0123456789"
    assert "0123456789" not in str(store.list_bookings()) + str(store.bookings)
    assert store.update_booking(booking["booking_id"], "SU124", "2027-06-12")["status"] == "updated"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT passport FROM bookings").fetchone()[0] == "0123456789"
    assert store.cancel_booking(booking["booking_id"])["status"] == "cancelled"
    store.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT passport FROM flight_store_cancelled_bookings").fetchone()[0] is None


def test_concurrent_connections_prevent_overselling_and_duplicates(tmp_path):
    path = tmp_path / "flights.sqlite3"
    stores = [FlightStore(path) for _ in range(4)]
    barrier = Barrier(len(stores))

    def book(item):
        index, store = item
        barrier.wait()
        return store.book("SU123", "2027-06-12", "Иван", f"{index:010d}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(book, enumerate(stores)))
    assert sorted(result["status"] for result in results) == ["booked", "sold_out", "sold_out", "sold_out"]
    for store in stores:
        store.close()

    path = tmp_path / "duplicate.sqlite3"
    stores = [FlightStore(path) for _ in range(4)]
    barrier = Barrier(len(stores))

    def duplicate(store):
        barrier.wait()
        return store.book("SU123", "2027-06-10", "Иван", "1234567890")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(duplicate, stores))
    assert sorted(result["status"] for result in results) == ["booked", "duplicate", "duplicate", "duplicate"]
    assert len({result["booking_id"] for result in results}) == 1
    assert stores[0].status("SU123", "2027-06-10")["flight"]["seats"] == 1
    for store in stores:
        store.close()


def test_empty_schedule_is_not_reseeded_after_initialization(tmp_path):
    path = tmp_path / "flights.sqlite3"
    store = FlightStore(path)
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM flights")
    store = FlightStore(path)
    assert store.search()["count"] == 0
    store.close()
