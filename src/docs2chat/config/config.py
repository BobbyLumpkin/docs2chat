"""
Purpose: Config class definition and instantiations.
"""


import os
from pathlib import Path
import yaml


class PathConcatenator(yaml.YAMLObject): 
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!osjoin"
    
    @classmethod
    def from_yaml(cls, loader, node): 
        seq = loader.construct_sequence(node)
        return os.path.join(*seq)


class StringConcatenator(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!strjoin"

    @classmethod
    def from_yaml(cls, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join(seq)


class ProjectPathConvertor(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!projpath"

    @classmethod
    def from_yaml(cls, loader, node):
        return Path(os.path.realpath(__file__)).parents[3].absolute()
            

class Config:
    def __init__(self, rel_path, bucket=None):
        self.base_path = Path(os.path.realpath(__file__)).parents[1].absolute()
        self.config_yaml = self.base_path / rel_path
        self.reset_config(rel_path)
        return
    
    def _process(self, **params): 
        for key, value in params.items(): 
            setattr(self, key, value)
    
    def reset_config(self, rel_path):
        with open(self.config_yaml, "rb") as f: 
            config = yaml.safe_load(f.read())
            self._process(**config)
            return


config = Config("config/config.yaml")