from flight_agent.data import FlightStore


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
