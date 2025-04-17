from accelerate import Accelerator


class AcceleratorSaveTrainableParams(Accelerator):
    def get_state_dict(self, model, unwrap=True):
        state_dict = super().get_state_dict(model, unwrap)
        if hasattr(model, "param_names_to_save"):
            param_names_to_save = model.param_names_to_save
            return {
                k: v
                for k, v in state_dict.items() if k in param_names_to_save
            }
        return state_dict
