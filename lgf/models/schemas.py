def get_schema(config):
    return add_normalization_layers(get_base_schema(config), config)


def get_base_schema(config):
    schema = []

    if config["logit_tf_lambda"] is not None:
        schema.append({
            "type": "logit",
            "lambda": config["logit_tf_lambda"],
            "scale": config["logit_tf_scale"]
        })


    model = config["model"]

    if model == "multiscale-realnvp":
        schema += get_multiscale_realnvp_schema(
            coupler_hidden_channels=config["g_nets"]["hidden_channels"],
        )

    elif model == "flat-realnvp":
        schema += get_flat_realnvp_schema(
            num_density_layers=config["num_density_layers"],
            coupler_hidden_units=config["g_nets"]["hidden_units"]
        )

    elif model == "maf":
        schema += get_maf_schema(
            num_density_layers=config["num_density_layers"],
            ar_map_hidden_units=config["g_nets"]["hidden_units"]
        )

    else:
        assert False, f"Invalid model {model}"

    return schema


def add_normalization_layers(base_schema, config):
    schema = []

    for layer in base_schema:
        if layer["type"] == "acl":
            layer["num_u_channels"] = 0

        schema.append(layer)

        if layer["type"] in ["acl", "made"]:
            if config["num_u_channels"] > 0:
                cond_affine = {
                    "type": "cond-affine",
                    "separate_coupler_nets": False,
                    "num_u_channels": config["num_u_channels"]
                }

                if config["model"] in ["flat-realnvp", "maf"]:
                    cond_affine = {
                        **cond_affine,
                        "st_net": {"type": "mlp", "hidden_units": config["st_nets"]["hidden_units"]},
                        "p_net": validate_net_config("mlp", config["p_nets"]),
                        "q_net": validate_net_config("mlp", config["q_nets"])
                    }

                elif config["model"] == "multiscale-realnvp":
                    cond_affine = {
                        **cond_affine,
                        "st_net": {"type": "resnet", "hidden_channels": config["st_nets"]["hidden_channels"]},
                        "p_net": validate_net_config("resnet", config["p_nets"]),
                        "q_net": validate_net_config("resnet", config["q_nets"])
                    }

                else:
                    assert False, f"Invalid model {config['model']}"

                schema.append(cond_affine)

            elif config["batch_norm"]:
                schema.append({"type": "batch-norm"})

    return schema


def validate_net_config(net_type, net_config):
    if net_type == "resnet":
        return {
            "type": "resnet",
            "hidden_channels": net_config["hidden_channels"]
        }
    elif net_type == "mlp":
        return {
            "type": "mlp",
            "hidden_units": net_config["hidden_units"]
        }
    else:
        assert False, f"Invalid net type {net_type}"


def get_multiscale_realnvp_schema(coupler_hidden_channels):
    schema = [
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "squeeze", "factor": 2},
        {"type": "acl", "mask_type": "split_channel", "reverse_mask": True},
        {"type": "acl", "mask_type": "split_channel", "reverse_mask": False},
        {"type": "acl", "mask_type": "split_channel", "reverse_mask": True},
        {"type": "split"},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True}
    ]

    for layer in schema:
        if layer["type"] == "acl":
            layer["separate_coupler_nets"] = False
            layer["coupler_net"] = {
                "type": "resnet",
                "hidden_channels": coupler_hidden_channels
            }

    return schema


def get_flat_realnvp_schema(
        num_density_layers,
        coupler_hidden_units
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result.append({
            "type": "acl",
            "mask_type": "alternating_channel",
            "reverse_mask": i % 2 == 0,
            "coupler_net": {
                "type": "mlp",
                "hidden_units": coupler_hidden_units
            },
            "separate_coupler_nets": True,
        })

    return result


def get_maf_schema(
        num_density_layers,
        ar_map_hidden_units
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result.append({
            "type": "made",
            "ar_map_hidden_units": ar_map_hidden_units
        })

    return result
