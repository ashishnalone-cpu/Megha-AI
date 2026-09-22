"""
geo_places.py

Wraps the GeoNames-based IndiaGeoAgent (as provided) to resolve city,
village, and landmark names to lat/lon. Loading IN.txt is the expensive
step, so build one IndiaGeoAgent per process/app session and reuse it
(wrap the constructor call in st.cache_resource in the app).
"""

import pandas as pd
from rapidfuzz import process, fuzz


class IndiaGeoAgent:
    def __init__(self, geonames_path="IN.txt", extra_csvs=None):
        cols = [
            "geonameid", "name", "asciiname", "alternatenames",
            "latitude", "longitude", "feature_class", "feature_code",
            "country_code", "cc2", "admin1", "admin2", "admin3", "admin4",
            "population", "elevation", "dem", "timezone", "modification_date",
        ]

        df = pd.read_csv(
            geonames_path, sep="\t", header=None, names=cols,
            dtype={"latitude": float, "longitude": float},
            low_memory=False, encoding="utf-8",
        )

        df = df[df["country_code"] == "IN"][
            ["name", "asciiname", "alternatenames", "latitude", "longitude",
             "feature_class", "feature_code", "admin1", "population"]
        ].copy()

        if extra_csvs:
            for csv in extra_csvs:
                extra = pd.read_csv(csv)
                df = pd.concat([df, extra], ignore_index=True)

        self.df = df.reset_index(drop=True)

        names = []
        self.name_to_idx = []
        for idx, row in self.df.iterrows():
            candidates = [row["name"], row["asciiname"]]
            if pd.notna(row["alternatenames"]):
                candidates += str(row["alternatenames"]).split(",")
            for n in candidates:
                n = str(n).strip()
                if n:
                    names.append(n)
                    self.name_to_idx.append(idx)

        self.names = names

    def geocode(self, query: str, limit: int = 5, min_score: int = 75):
        if not self.names:
            return []
        matches = process.extract(
            query, self.names, scorer=fuzz.WRatio, limit=limit * 5
        )

        results = []
        seen = set()
        for name, score, list_idx in matches:
            if score < min_score:
                continue
            idx = self.name_to_idx[list_idx]
            row = self.df.iloc[idx]
            key = (round(row["latitude"], 5), round(row["longitude"], 5))
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "name": row["name"],
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "feature": row.get("feature_class", ""),
                "code": row.get("feature_code", ""),
                "state_code": row.get("admin1", ""),
                "population": row.get("population", 0),
                "score": score,
            })
            if len(results) >= limit:
                break
        return results

    def get_lat_lon(self, place: str):
        res = self.geocode(place, limit=1)
        if res:
            return res[0]["lat"], res[0]["lon"]
        return None
