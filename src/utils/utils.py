def test():
    print("This is a test function in utils.py")


def get_features_by_dtype(df):
    """Returns dict of features by their data types."""
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    datetime_features = df.select_dtypes(include=["datetime64"]).columns.tolist()

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "datetime_features": datetime_features,
    }


def get_features_by_type(df):
    """Returns dict of features by their types: temporal, station, spatial."""
    temporal_data = [
        "temperatures",
        "precipitations",
        "evaporation",
        "soil_moisture",
        "water_flow_week1",
        "water_flow_week2",
        "water_flow_week3",
        "water_flow_week4",
    ]

    # Station data
    station_data = [
        "ObsDate",
        "station_code",
        "latitude",
        "longitude",
        "altitude",
        "area",
        "catchment",
        "north_hemisphere",
    ]

    # Spatial data: Bulk density, Coarse fragments, Proportion of clay, Proportion of sand, and other static attributes
    spatial_data = [
        col
        for col in df.columns
        if col not in temporal_data and col not in station_data
    ]

    # Create the dictionary
    data_dict = {
        "temporal_data": temporal_data,
        "station_data": station_data,
        "spatial_data": spatial_data,
    }

    return data_dict
