from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
import timm
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

# AutoConfig.register("resnet", ResnetConfig)
# AutoModel.register(ResnetConfig, ResnetModel)
# AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)

ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")

resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())

resnet50d.push_to_hub("custom-resnet50d")