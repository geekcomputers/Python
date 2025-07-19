def resistance_calculator(
    material: str, length: float, section: float, temperature: float
) -> str:
    """
    Calculate the electrical resistance of a wire based on its properties.

    Parameters:
        material (str): Material of the wire (available: silver, copper, aluminium, tungsten, iron, steel, zinc, solder)
        length (float): Length of the wire in meters
        section (float): Cross-sectional area of the wire in square millimeters (mm²)
        temperature (float): Operating temperature of the wire in °C

    Returns:
        str: Calculated resistance in ohms (Ω)
    """
    # Material properties: resistivity at 20°C (ρ in Ω·mm²/m) and temperature coefficient (α in 1/°C)
    materials: dict[str, dict[str, float]] = {
        "silver": {"rho": 0.0163, "coefficient": 0.0038},
        "copper": {"rho": 0.0178, "coefficient": 0.00381},
        "aluminium": {"rho": 0.0284, "coefficient": 0.004},
        "tungsten": {"rho": 0.055, "coefficient": 0.0045},
        "iron": {"rho": 0.098, "coefficient": 0.006},
        "steel": {"rho": 0.15, "coefficient": 0.0047},
        "zinc": {"rho": 0.06, "coefficient": 0.0037},
        "solder": {"rho": 0.12, "coefficient": 0.0043},
    }

    # Get material properties (will raise KeyError for invalid materials)
    rho_20deg: float = materials[material]["rho"]
    temp_coefficient: float = materials[material]["coefficient"]

    # Calculate resistivity at operating temperature
    rho: float = rho_20deg * (1 + temp_coefficient * (temperature - 20))

    # Calculate resistance (R = ρ * L / S)
    resistance: float = rho * length / section

    return f"{resistance}Ω"
