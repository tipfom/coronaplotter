MAINLAND_CHINA = 0
WESTERN_PACIFIC_REGION = 1
EUROPEAN_REGION = 2
SOUTH_EAST_ASIA_REGION = 3
EASTERN_MEDITERRANEAN_REGION = 4
REGION_OF_THE_AMERICANS = 5
AFRICAN_REGION = 6
OTHER = 7

REGION_COUNT = OTHER + 1

region_map = {
    "Mainland China": MAINLAND_CHINA,
    "Hong Kong": MAINLAND_CHINA,
    "Macau": MAINLAND_CHINA,
    "Taiwan": MAINLAND_CHINA,
    #####################################################################
    "South Korea": WESTERN_PACIFIC_REGION,
    "Japan": WESTERN_PACIFIC_REGION,
    "Singapore": WESTERN_PACIFIC_REGION,
    "Australia": WESTERN_PACIFIC_REGION,
    "Malaysia": WESTERN_PACIFIC_REGION,
    "Vietnam": WESTERN_PACIFIC_REGION,
    "Philippines": WESTERN_PACIFIC_REGION,
    "Cambodia": WESTERN_PACIFIC_REGION,
    "New Zealand": WESTERN_PACIFIC_REGION,
    #####################################################################
    "Italy": EUROPEAN_REGION,
    "France": EUROPEAN_REGION,
    "Germany": EUROPEAN_REGION,
    "Spain": EUROPEAN_REGION,
    "UK": EUROPEAN_REGION,
    "Switzerland": EUROPEAN_REGION,
    "Norway": EUROPEAN_REGION,
    "Sweden": EUROPEAN_REGION,
    "Austria": EUROPEAN_REGION,
    "Croatia": EUROPEAN_REGION,
    "Netherlands": EUROPEAN_REGION,
    "Azerbaijan": EUROPEAN_REGION,
    "Denmark": EUROPEAN_REGION,
    "Georgia": EUROPEAN_REGION,
    "Greece": EUROPEAN_REGION,
    "Romania": EUROPEAN_REGION,
    "Finland": EUROPEAN_REGION,
    "Russia": EUROPEAN_REGION,
    "Belarus": EUROPEAN_REGION,
    "Belgium": EUROPEAN_REGION,
    "Estonia": EUROPEAN_REGION,
    "Ireland": EUROPEAN_REGION,
    "Lithuania": EUROPEAN_REGION,
    "Monaco": EUROPEAN_REGION,
    "North Macedonia": EUROPEAN_REGION,
    "San Marino": EUROPEAN_REGION,
    "Luxembourg": EUROPEAN_REGION,
    "Iceland": EUROPEAN_REGION,
    "Czech Republic": EUROPEAN_REGION,
    "Andorra": EUROPEAN_REGION,
    "Portugal": EUROPEAN_REGION,
    "Latvia": EUROPEAN_REGION,
    "Ukraine": EUROPEAN_REGION,
    "Hungary": EUROPEAN_REGION,
    "Gibraltar": EUROPEAN_REGION,  # ????????????????
    "Faroe Islands": EUROPEAN_REGION,
    "Liechtenstein": EUROPEAN_REGION,
    "Poland": EUROPEAN_REGION,
    "Bosnia and Herzegovina": EUROPEAN_REGION,
    "Slovenia": EUROPEAN_REGION,
    "Serbia": EUROPEAN_REGION,
    "Slovakia": EUROPEAN_REGION,
    "Vatican City": EUROPEAN_REGION,
    "Malta": EUROPEAN_REGION,
    "Republic of Ireland": EUROPEAN_REGION,
    "Bulgaria": EUROPEAN_REGION,
    "Moldova": EUROPEAN_REGION,
    "Albania": EUROPEAN_REGION,
    "Cyprus": EUROPEAN_REGION,
    #####################################################################
    "Thailand": SOUTH_EAST_ASIA_REGION,
    "Indonesia": SOUTH_EAST_ASIA_REGION,
    "India": SOUTH_EAST_ASIA_REGION,
    "Nepal": SOUTH_EAST_ASIA_REGION,
    "Sri Lanka": SOUTH_EAST_ASIA_REGION,
    "Bhutan": SOUTH_EAST_ASIA_REGION,
    "Maldives": SOUTH_EAST_ASIA_REGION,
    "Bangladesh": SOUTH_EAST_ASIA_REGION,
    "Brunei": SOUTH_EAST_ASIA_REGION,
    #####################################################################
    "Armenia": EASTERN_MEDITERRANEAN_REGION,  # ????????????
    "Iran": EASTERN_MEDITERRANEAN_REGION,
    "Kuwait": EASTERN_MEDITERRANEAN_REGION,
    "Bahrain": EASTERN_MEDITERRANEAN_REGION,
    "United Arab Emirates": EASTERN_MEDITERRANEAN_REGION,
    "Iraq": EASTERN_MEDITERRANEAN_REGION,
    "Oman": EASTERN_MEDITERRANEAN_REGION,
    "Pakistan": EASTERN_MEDITERRANEAN_REGION,
    "Lebanon": EASTERN_MEDITERRANEAN_REGION,
    "Afghanistan": EASTERN_MEDITERRANEAN_REGION,
    "Egypt": EASTERN_MEDITERRANEAN_REGION,
    "Qatar": EASTERN_MEDITERRANEAN_REGION,
    "Saudi Arabia": EASTERN_MEDITERRANEAN_REGION,
    "Jordan": EASTERN_MEDITERRANEAN_REGION,  # ??????????????
    "Israel": EASTERN_MEDITERRANEAN_REGION,
    "Palestine": EASTERN_MEDITERRANEAN_REGION,
    #####################################################################
    "US": REGION_OF_THE_AMERICANS,
    "Canada": REGION_OF_THE_AMERICANS,
    "Brazil": REGION_OF_THE_AMERICANS,
    "Mexico": REGION_OF_THE_AMERICANS,
    "Ecuador": REGION_OF_THE_AMERICANS,
    "Dominican Republic": REGION_OF_THE_AMERICANS,  # ????????????
    "Chile": REGION_OF_THE_AMERICANS,  # ?????????????????
    "Argentina": REGION_OF_THE_AMERICANS,  # ????????????
    "Saint Barthelemy": REGION_OF_THE_AMERICANS,  # ?????????????
    "Peru": REGION_OF_THE_AMERICANS,
    "Martinique": REGION_OF_THE_AMERICANS,  # ?????????????
    "Colombia": REGION_OF_THE_AMERICANS,
    "Costa Rica": REGION_OF_THE_AMERICANS,
    "Paraguay": REGION_OF_THE_AMERICANS,
    "St. Martin": REGION_OF_THE_AMERICANS,  # ??????????????
    #####################################################################
    "Algeria": AFRICAN_REGION,
    "Nigeria": AFRICAN_REGION,
    "Morocco": AFRICAN_REGION,
    "Senegal": AFRICAN_REGION,
    "Tunisia": AFRICAN_REGION,
    "South Africa": AFRICAN_REGION,
    "Togo": AFRICAN_REGION,
    "French Guiana": AFRICAN_REGION,
    "Cameroon": AFRICAN_REGION,
    #####################################################################
    "Others": OTHER,
}
