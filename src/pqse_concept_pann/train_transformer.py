from pqse_concept_pann.experiments import exp_transformer_config, load_data, exp_pann_config


if __name__ == "__main__":
    # Train Transformer
    network_data = load_data()
    base_model_configs = [exp_transformer_config()]
    for cfg in base_model_configs:
        hyperparams, callbacks, model_class, custom_layer, exp_path, label = cfg
        m = model_class(hyperparams, network_data, callbacks)
        model = m.create_model()
        model.summary()
        m.train(model, "transformer/")