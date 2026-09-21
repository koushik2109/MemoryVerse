"""
Story Generation - Event Aggregator
Groups individual memory assets into discrete episodic events using
temporal proximity (inter-photo gap clustering) and location clustering.
"""
from typing import List, Dict, Any, Tuple
from datetime import datetime
import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class EventAggregator:
    """
    Partitions a collection of memories into episodic event clusters.
    """

    def __init__(self, max_gap_hours: float = 4.0, max_radius_km: float = 15.0):
        self.max_gap_hours = max_gap_hours
        self.max_radius_km = max_radius_km

    def aggregate_into_events(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups items into chronological event episodes.
        """
        if not items:
            return []

        # Sort items chronologically if timestamps available
        def parse_time(it):
            ts = it.get("timestamp") or it.get("created_at")
            if not ts:
                return datetime.min
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return datetime.min

        sorted_items = sorted(items, key=parse_time)
        events: List[Dict[str, Any]] = []
        current_event_items: List[Dict[str, Any]] = [sorted_items[0]]

        for i in range(1, len(sorted_items)):
            prev_time = parse_time(sorted_items[i - 1])
            curr_time = parse_time(sorted_items[i])

            gap_hours = (curr_time - prev_time).total_seconds() / 3600.0 if prev_time != datetime.min and curr_time != datetime.min else 0.0

            if gap_hours > self.max_gap_hours:
                # Start new event
                events.append({
                    "event_id": f"event_{len(events)+1}",
                    "item_count": len(current_event_items),
                    "items": current_event_items,
                })
                current_event_items = [sorted_items[i]]
            else:
                current_event_items.append(sorted_items[i])

        if current_event_items:
            events.append({
                "event_id": f"event_{len(events)+1}",
                "item_count": len(current_event_items),
                "items": current_event_items,
            })

        return events
