from app.core.db import get_supabase_client
from app.schemas.domain import (
    TimelineResponse, TimelineYearGroup, TimelineMonthGroup,
    TimelineDayGroup, MemoryResponse
)
from typing import cast, Any
from datetime import datetime, timezone
from collections import defaultdict
import calendar


class TimelineService:
    """Groups user's memories chronologically into year → month → day buckets."""

    @staticmethod
    def get_timeline(user_id: str, vault_id: str | None = None) -> TimelineResponse:
        supabase = get_supabase_client()

        # Fetch memories (with joined media if possible, or fetch separately if Supabase Python client struggles with nested queries)
        # Actually, the python client can do `select("*, media(*)")` but we need to map to MemoryResponse.
        query = (
            supabase.table("memories")
            .select("*, media!media_memory_id_fkey(*)")
            .eq("owner_id", user_id)
            .order("memory_date", desc=True)
            .limit(500)
        )
        if vault_id:
            query = query.eq("vault_id", vault_id)

        res = query.execute()
        memories_list = cast(list[dict[str, Any]], res.data or [])

        if not memories_list:
            return TimelineResponse(groups=[], total_items=0)

        # ── Bucket by ISO year → calendar-month → day ─────────────
        # Structure: {year: {month_num: {date: [MemoryResponse]}}}
        year_month_day: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        total_items = len(memories_list)

        for m in memories_list:
            dt = TimelineService._parse_dt(m.get("memory_date"))
            if dt is None:
                continue
                
            # Create MemoryResponse
            media_arr = m.get("media", [])
            media_count = len(media_arr)
            
            # Sort media by created_at desc internally
            try:
                media_arr.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            except Exception:
                pass

            m_response = MemoryResponse(
                id=m["id"],
                owner_id=m["owner_id"],
                vault_id=m.get("vault_id"),
                title=m["title"],
                description=m.get("description"),
                cover_media_id=m.get("cover_media_id"),
                memory_date=dt,
                location_name=m.get("location_name"),
                created_at=TimelineService._parse_dt(m.get("created_at")) or datetime.now(timezone.utc),
                updated_at=TimelineService._parse_dt(m.get("updated_at")) or datetime.now(timezone.utc)
            )
            # Python models don't currently have media inside MemoryResponse directly on the timeline schema unless we add it,
            # but wait, `MemoryResponse` doesn't strictly have `media` array according to Pydantic if we didn't add it to domain.py?
            # Let's check `MemoryResponse` in domain.py: it does not have `media` list defined in `MemoryResponse`! It's added dynamically as `dict` in `memories.py`.
            # Wait, `MemoryResponse` in `domain.py` DOES NOT have `media: List[...]` ?
            # If so, returning it via Pydantic will strip `media` if it's not defined!
            # Let's add it via dict or update domain.py again. Actually, I can just use a dictionary if Pydantic allows `extra="allow"` or I can just rely on the frontend handling.
            # I will just create a dict that matches the JSON the frontend expects, because Pydantic `dict()` can be bypassed if we just return dicts.
            
            year = dt.year
            month = dt.month
            day_date = datetime(year, month, dt.day)

            # We'll store a dict that conforms to the JSON output expected
            mem_dict = m_response.dict()
            mem_dict['media'] = media_arr
            mem_dict['media_count'] = media_count
            
            year_month_day[year][month][day_date].append(mem_dict)

        # ── Build the response ────────────────────────────────────────────────
        year_groups: list[TimelineYearGroup] = []

        for year in sorted(year_month_day.keys(), reverse=True):
            month_groups: list[TimelineMonthGroup] = []

            for month_num in sorted(year_month_day[year].keys(), reverse=True):
                day_groups: list[TimelineDayGroup] = []

                for day_date in sorted(year_month_day[year][month_num].keys(), reverse=True):
                    items = year_month_day[year][month_num][day_date]
                    
                    day_label = f"{calendar.month_abbr[day_date.month]} {day_date.day}"

                    day_groups.append(TimelineDayGroup(
                        date_label=day_label,
                        date=day_date,
                        memories=items
                    ))

                month_name = calendar.month_name[month_num]
                month_groups.append(TimelineMonthGroup(month=month_name, days=day_groups))

            year_groups.append(TimelineYearGroup(year=str(year), months=month_groups))

        return TimelineResponse(groups=year_groups, total_items=total_items)

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            clean = value.replace("Z", "+00:00")
            return datetime.fromisoformat(clean).replace(tzinfo=None)
        except Exception:
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None
