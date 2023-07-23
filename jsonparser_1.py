import json


class JsonParser:
    """
    this class to handle anything related to json file [as implementation of facade pattern]
    """

    def convert_json_to_python(self, par_json_file):
        """
        this function to convert any json file format to dictionary
        args: the json file
        return: dictionary contains json file data
        """
        with open(par_json_file) as json_file:
            data_dic = json.load(json_file)
        return data_dic

    def convert_python_to_json(self, par_data_dic, par_json_file=""):
        """
        this function converts dictionary of data to json string and store it in json file if
        json file pass provided if not it only returns the json string
        args:
             par_data_dic: dictionary of data
             par_json_file: the output json file
        return: json string
        """
        if par_json_file:
            with open(par_json_file, "w") as outfile:
                return json.dump(par_data_dic, outfile)
        else:
            return json.dump(par_data_dic)

    def get_json_value(self, par_value, par_json_file):
        """
        this function gets specific dictionary key value from json file
        args:
             par_value: dictionary key value
             par_json_file: json file
             return: value result
        """
        data_dic = self.convert_json_to_python(par_json_file)
        return data_dic[par_value]
