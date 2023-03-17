def resistance_calculator(material:str, lenght:float, section:float, temperature:float):
  """
  material is a string indicating the material of the wire
  
  lenght is a floating value indicating the lenght of the wire in meters

  diameter is a floating value indicating the diameter of the wire in millimeters

  temperature is a floating value indicating the temperature at which the wire is operating in °C

  Available materials:
  - silver
  - copper
  - aluminium
  - tungsten
  - iron
  - steel
  - zinc
  - solder"""

  materials = {
    "silver": {
                "rho": 0.0163,
                "coefficient": 0.0038
              },

    "copper": {
                "rho": 0.0178,
                "coefficient": 0.00381
              },

    "aluminium": {
                  "rho": 0.0284,
                  "coefficient": 0.004
                 },

    "tungsten": {
                  "rho": 0.055,
                  "coefficient": 0.0045
                },

    "iron": {
              "rho": 0.098,
              "coefficient": 0.006
            },

    "steel": {
                "rho": 0.15,
                "coefficient": 0.0047
              },

    "zinc": {
              "rho": 0.06,
              "coefficient": 0.0037
            },

    "solder": {
                "rho": 0.12,
                "coefficient": 0.0043
              }
  }

  rho_20deg = materials[material]["rho"]
  temp_coefficient = materials[material]["coefficient"]

  rho = rho_20deg * (1 + temp_coefficient * (temperature - 20))
  resistance = rho * lenght / section

  return f"{resistance}Ω"
