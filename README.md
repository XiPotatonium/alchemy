# alchemy

希望通过模块化设计将项目编写简化为编写模块+修改配置文件。

[Wiki](https://github.com/XiPotatonium/alchemy/wiki)

## Related repos

* [alchemy-web](https://github.com/XiPotatonium/alchemy-web).
* [alchemy-ner](https://github.com/XiPotatonium/alchemy-ner), a project based on alchemy, where we implement the named entity recognition (NER) task. We also implement several baselines of the NER task. This repository can be used as a template if you want to implement your own new NER model. You can reuse the code for dataload, training, evaluation, etc.

## Dependencies

conda:

* rich
* loguru
* typer
* pytorch
* numpy
* tomlkit (optional if you want to use other format)
* tensorboard (optional if you don't use tensorboard)

pip:

* fastapi (optional if you don't use our web util)
* uvicorn (optional if you don't use our web util)
* pynvml (optional if you don't use our device allocation util)

## How to use as a submodule

1. Add this repo as a submodule to your project.

```bash
git submodule add git@github.com:XiPotatonium/alchemy.git
```

2. Download the submodule.

```bash
git submodule update --init --recursive
```

3. Update the submodule

```bash
cd alchemy
git pull
```
