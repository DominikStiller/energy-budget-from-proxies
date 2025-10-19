from __future__ import annotations

# Area from https://github.com/ClimateIndicator/ocean-heat-content/blob/main/Code/plot_IPCC_update_EnergyInventory_Figure.ipynb
SURFACE_AREA_EARTH = 510e12  # , m^2, 510 million km^2
SURFACE_AREA_OCEAN = 361e12  # m^2, 361 million km^2
OCEAN_FRACTION = SURFACE_AREA_OCEAN / SURFACE_AREA_EARTH  # 0.707

EARTH_RADIUS_MEAN = 6371.0088e3  # m

SECONDS_PER_DAY = 86400
DAYS_PER_YEAR = 365.2422
SECONDS_PER_YEAR = SECONDS_PER_DAY * DAYS_PER_YEAR

OCEAN_HEAT_UPTAKE_FRACTION = 0.90  # Von Schuckmann et al. (2023)
