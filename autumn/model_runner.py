from numpy import isfinite
import copy

def is_positive_definite(v):
    return isfinite(v) and v > 0.0

class ModelRunner:

    def __init__(self,
                 model=None):
        self.model = copy.deepcopy(model)
        self.is_last_run_success = False
        self.nb_accepted = 0

    def set_model_with_params(self, param_dict):
        n_set = 0
        for key in param_dict:
            if key in self.model.params:
                n_set += 1
                self.model.set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model params" % key)

    def convert_param_list_to_dict(self, params):
        param_dict = {}
        for val, props in zip(params, self.model.param_ranges_unc):
            param_dict[props['key']] = val
        return param_dict

    def run_with_params(self, params):
        for i, p in enumerate(params):
            if not is_positive_definite(p):
                print "Warning: parameter%d=%f is invalid for model" % (i, p)
                self.is_last_run_success = False
                return
            bounds = self.model.param_ranges_unc[i]["bounds"]
            if (p < bounds[0]) or (p > bounds[1]):
                #print "Warning: parameter%d=%f is outside of the allowed bounds" % (i, p)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        try:
            self.model.integrate()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False