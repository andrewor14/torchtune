.. _models:

================
torchtune.models
================

.. currentmodule:: torchtune.models

llama3
------

All models from the `Llama3 family <https://llama.meta.com/llama3/>`_.

.. code-block:: bash

    tune download meta-llama/Meta-Llama-3-8B-Instruct --hf-token <ACCESS_TOKEN>


.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama3.llama3_8b
    llama3.lora_llama3_8b
    llama3.qlora_llama3_8b
    llama3.llama3_70b
    llama3.lora_llama3_70b
    llama3.qlora_llama3_70b
    llama3.llama3_tokenizer
    llama3.Llama3Tokenizer


llama2
------

All models from the `Llama2 family <https://llama.meta.com/llama2/>`_.

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download meta-llama/Llama-2-7b-hf --hf-token <ACCESS_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    llama2.llama2_7b
    llama2.lora_llama2_7b
    llama2.qlora_llama2_7b
    llama2.llama2_13b
    llama2.lora_llama2_13b
    llama2.qlora_llama2_13b
    llama2.llama2_70b
    llama2.lora_llama2_70b
    llama2.qlora_llama2_70b
    llama2.llama2_tokenizer
    llama2.Llama2Tokenizer


code llama
----------

Models from the `Code Llama family <https://arxiv.org/pdf/2308.12950>`_.

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download codellama/CodeLlama-7b-hf --hf-token <ACCESS_TOKEN>

.. autosummary::
    :toctree: generated/
    :nosignatures:

    code_llama2.code_llama2_7b
    code_llama2.lora_code_llama2_7b
    code_llama2.qlora_code_llama2_7b
    code_llama2.code_llama2_13b
    code_llama2.lora_code_llama2_13b
    code_llama2.qlora_code_llama2_13b
    code_llama2.code_llama2_70b
    code_llama2.lora_code_llama2_70b
    code_llama2.qlora_code_llama2_70b


phi-3
-----

Models from the `Phi-3 mini family <https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/>`_.

Pre-trained models can be download from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download microsoft/Phi-3-mini-4k-instruct --hf-token <HF_TOKEN> --ignore-patterns ""

.. autosummary::
    :toctree: generated/
    :nosignatures:

    phi3.phi3_mini
    phi3.lora_phi3_mini
    phi3.qlora_phi3_mini
    phi3.phi3_mini_tokenizer
    phi3.Phi3MiniTokenizer


mistral
-------

All models from `Mistral AI family <https://mistral.ai/technology/#models>`_.

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download mistralai/Mistral-7B-v0.1

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mistral.mistral_7b
    mistral.lora_mistral_7b
    mistral.qlora_mistral_7b
    mistral.mistral_classifier_7b
    mistral.lora_mistral_classifier_7b
    mistral.qlora_mistral_classifier_7b
    mistral.mistral_tokenizer
    mistral.MistralTokenizer


gemma
-----

Models of size 2B and 7B from the `Gemma family <https://blog.google/technology/developers/gemma-open-models/>`_.

Pre-trained models can be downloaded from the Hugging Face Hub with the following command:

.. code-block:: bash

    tune download google/gemma-2b --hf-token <ACCESS_TOKEN> --ignore-patterns ""

.. autosummary::
    :toctree: generated/
    :nosignatures:

    gemma.gemma_2b
    gemma.lora_gemma_2b
    gemma.qlora_gemma_2b
    gemma.gemma_7b
    gemma.lora_gemma_7b
    gemma.qlora_gemma_7b
    gemma.gemma_tokenizer
    gemma.GemmaTokenizer


clip
-----

Vision components to support multimodality using `CLIP encoder <https://arxiv.org/abs/2103.00020>`_.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    clip.clip_vision_encoder
    clip.TokenPositionalEmbedding
    clip.TiledTokenPositionalEmbedding
    clip.TilePositionalEmbedding
