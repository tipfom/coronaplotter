MAINLAND_CHINA = 0
WESTERN_PACIFIC_REGION = 1
EUROPEAN_REGION = 2
SOUTH_EAST_ASIA_REGION = 3
EASTERN_MEDITERRANEAN_REGION = 4
REGION_OF_THE_AMERICANS = 5
AFRICAN_REGION = 6
SOUTH_KOREA_COUNTRY = 7
ITALY_COUNTRY = 8
IRAN_COUNTRY = 9
FRANCE_COUNTRY = 10
GERMANY_COUNTY = 11
OTHER = 12

REGION_COUNT = OTHER + 1

country_map = {
    "Korea, South": SOUTH_KOREA_COUNTRY,
    "Italy": ITALY_COUNTRY,
    "France": FRANCE_COUNTRY,
    "Germany": GERMANY_COUNTY,
    "Iran": IRAN_COUNTRY,
}

region_map = {
    "China": MAINLAND_CHINA,
    "Hong Kong": MAINLAND_CHINA,
    "Macau": MAINLAND_CHINA,
    "Taiwan*": MAINLAND_CHINA,
    #####################################################################
    "Korea, South": WESTERN_PACIFIC_REGION,
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
    "United Kingdom": EUROPEAN_REGION,
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
    "Czechia": EUROPEAN_REGION,
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
    "Turkey": EUROPEAN_REGION,  # ?????????????????????
    "Holy See": EUROPEAN_REGION,
    "Jersey": EUROPEAN_REGION,
    "Guernsey": EUROPEAN_REGION,
    "Kosovo": EUROPEAN_REGION,
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
    "Mongolia": SOUTH_EAST_ASIA_REGION,  # ??????????
    "Uzbekistan": SOUTH_EAST_ASIA_REGION, # ?????????
    "Kazakhstan": SOUTH_EAST_ASIA_REGION,  # ????????????
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
    "occupied Palestinian territory": EASTERN_MEDITERRANEAN_REGION,
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
    "Honduras": REGION_OF_THE_AMERICANS,
    "Jamaica": REGION_OF_THE_AMERICANS,
    "Cuba": REGION_OF_THE_AMERICANS,
    "Guyana": REGION_OF_THE_AMERICANS,
    "Panama": REGION_OF_THE_AMERICANS,
    "Bolivia": REGION_OF_THE_AMERICANS,
    "Venezuela": REGION_OF_THE_AMERICANS,
    "Curacao": REGION_OF_THE_AMERICANS,  # ?????
    "Guatemala": REGION_OF_THE_AMERICANS,
    "Saint Lucia": REGION_OF_THE_AMERICANS,
    "Saint Vincent and the Grenadines": REGION_OF_THE_AMERICANS,
    "Aruba": REGION_OF_THE_AMERICANS,
    "Antigua and Barbuda": REGION_OF_THE_AMERICANS,
    "Uruguay": REGION_OF_THE_AMERICANS,
    "Trinidad and Tobago": REGION_OF_THE_AMERICANS,
    "Cayman Islands": REGION_OF_THE_AMERICANS,
    "Guadeloupe": REGION_OF_THE_AMERICANS,
    "Suriname": REGION_OF_THE_AMERICANS,
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
    "Congo (Kinshasa)": AFRICAN_REGION,
    "Cote d'Ivoire": AFRICAN_REGION,
    "Burkina Faso": AFRICAN_REGION,
    "Reunion": AFRICAN_REGION,
    "Ghana": AFRICAN_REGION,
    "Namibia": AFRICAN_REGION,
    "Seychelles": AFRICAN_REGION,
    "Eswatini": AFRICAN_REGION,
    "Gabon": AFRICAN_REGION,
    "Mauritania": AFRICAN_REGION,
    "Rwanda": AFRICAN_REGION,
    "Sudan": AFRICAN_REGION,
    "Kenya": AFRICAN_REGION,
    "Guinea": AFRICAN_REGION,
    "Congo (Brazzaville)": AFRICAN_REGION,
    "Equatorial Guinea": AFRICAN_REGION,
    "Central African Republic": AFRICAN_REGION,
    "Ethiopia": AFRICAN_REGION,
    #####################################################################
    "Others": OTHER,
    "Cruise Ship": OTHER,
}

region_names = {
    MAINLAND_CHINA: "China",
    WESTERN_PACIFIC_REGION: "Western Pacific Countries",
    EUROPEAN_REGION: "Europe",
    SOUTH_EAST_ASIA_REGION: "South East Asia",
    EASTERN_MEDITERRANEAN_REGION: "Eastern Mediteranian Countries",
    REGION_OF_THE_AMERICANS: "America",
    AFRICAN_REGION: "Africa",
    OTHER: "Other",
    SOUTH_KOREA_COUNTRY: "South Korea",
    ITALY_COUNTRY: "Italy",
    IRAN_COUNTRY: "Iran",
    FRANCE_COUNTRY: "France",
    GERMANY_COUNTY: "Germany",
}
