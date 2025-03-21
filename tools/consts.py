import torch

FEATURES = ['FuelGasGTYConsumption', 'FuelGasKYConsumption', 'FuelGasTemperature',
            'AirFurnaceConsumption', 'GasTurbineTemperature', 'EntranceKYTemperature',
            'AfterAfterburningKYTemperature', 'GasTurbineFrontPressure',
            'CombustionChamberTemperature', 'GasTurbinePower', 'VNAPosition',
            'AirAtmosphericTemperature', 'AirAtmosphericPressure']

TARGETS = ['FlueGasTemperature', 'FlueGasPressure', 'FlueGasConsumption',
           'ConcentrationO2', 'FlueGasHumidity', 'ConcentrationCO',
           'ConcentrationCO2', 'ConcentrationNOx']

limits = {
    "FlueGasTemperature": (0, 150),
    "FlueGasPressure": (-5.5, 5),
    # "FlueGasConsumption": (0, 3.35),  # пупупуууу
    "ConcentrationO2": (13, 22),
    "FlueGasHumidity": (0, 350),
    "ConcentrationCO": (0, 1200),
    "ConcentrationCO2": (0, 5),
    "ConcentrationNOx": (0, 1200),

    # "FuelGasGTYConsumption": (0, 10),
    # "FuelGasKYConsumption": (0, 1400),
    # "FuelGasTemperature": (-20, 40),
    # "AirFurnaceConsumption": (0, 600),  # пупупуууу
    # "GasTurbineTemperature": (-30, 40),  # пупупуууу
    # "EntranceKYTemperature": (0, 600),
    # "AfterAfterburningKYTemperature": (0, 600),
    # "GasTurbineFrontPressure": (0, 17),
    # "CombustionChamberTemperature": (0, 1250),
    # "GasTurbinePower": (0, 200),
    # "VNAPosition": (-65, 0),
    # "AirAtmosphericTemperature": (-40, 40),
    # "AirAtmosphericPressure": (900, 1050),
    # "AirAtmosphericHumidity": (0, 100)
}

ws = torch.tensor([1.63778039, 1.20387111, 1.20317918, 1.32377653, 22.05927316,
                   23.90044874, 97.1838969, 1.09202928])
