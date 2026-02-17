def get_data_ready(mixing_data):
    """This function gets data ready for modeling."""
    mixing_data.sort_values(
        by=["Fecha de Corte", "Horario.1"],
        inplace=True)
    mixing_data.reset_index(inplace=True, drop=True)
    return mixing_data


def modeling(mixing_data, SMC):
    """This function executes training."""
    mixing_data = get_data_ready(mixing_data)
    mixing_data = SMC.create_vars(mixing_data)
    model = SMC.train_test(mixing_data)
    SMC.save_model(model, mixing_data)
    return
