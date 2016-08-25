from numpy import isfinite

def is_positive_definite(v):
    return isfinite(v) and v > 0.0

class ModelRunner:
    def __init__(self,
                 model = None):

        self.model = model
        self.mode = 'uncertainty'
        self.is_last_run_success = False

        self.param_ranges_unc = self.model.param_ranges_unc
        self.outputs_unc = self.model.outputs_unc

        for key, value in self.model.inputs.model_constants.items():
            if type(value) == float:
                self.model.set_parameter(key, value)

        if self.mode == 'calibration':
            for props in self.model.param_props_list:
                self.model.set_parameter(props['key'], props['init'])

        self.data_to_fit = {}
        self.get_data_to_fit() # collect the data regarding incidence , mortality, etc. from the model object
        self.best_fit = {}
        self.nb_accepted = 0

    def get_data_to_fit(self):
        if self.mode == 'calibration':
            var_to_iterate = self.model.calib_outputs # for calibration
        elif self.mode == 'uncertainty':
            var_to_iterate = self.model.outputs_unc

        for output in var_to_iterate:
            if (output['key']) == 'incidence':
                self.data_to_fit['incidence'] = self.model.inputs.original_data['tb']['e_inc_100k']
                self.data_to_fit['incidence_low'] = self.model.inputs.original_data['tb']['e_inc_100k_lo']
                self.data_to_fit['incidence_high'] = self.model.inputs.original_data['tb']['e_inc_100k_hi']
            elif (output['key']) == 'mortality':
                self.data_to_fit['mortality'] = self.model.inputs.original_data['tb']['e_mort_exc_tbhiv_100k']
                self.data_to_fit['mortality_low'] = self.model.inputs.original_data['tb']['e_mort_exc_tbhiv_100k_lo']
                self.data_to_fit['mortality_high'] = self.model.inputs.original_data['tb']['e_mort_exc_tbhiv_100k_hi']
            else:
                print "Warning: Calibrated output %s is not directly available from the data" % output['key']

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

    # define the characteristics of the normal distribution for model outputs (incidence, mortality)
    def get_normal_char(self):
        normal_char = {}  # store the characteristics of the normal distributions
        for output_dict in self.model.outputs_unc:
            normal_char[output_dict['key']] = {}
            if output_dict['key'] == 'mortality':
                sd = output_dict['posterior_width'] / (2.0 * 1.96)
                for year in self.data_to_fit[output_dict['key']].keys():
                    mu = self.data_to_fit[output_dict['key']][year]
                    normal_char[output_dict['key']][year] = [mu, sd]

            elif output_dict['key'] == 'incidence':
                for year in self.data_to_fit[output_dict['key']].keys():
                    low = self.data_to_fit['incidence_low'][year]
                    high = self.data_to_fit['incidence_high'][year]
                    sd = (high - low) / (2.0 * 1.96)
                    mu = 0.5 * (high + low)
                    normal_char[output_dict['key']][year] = [mu, sd]

        return normal_char

