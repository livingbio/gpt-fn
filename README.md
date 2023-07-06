# GPT-Fn

[![CI](https://github.com/livingbio/gpt-fn/workflows/python-unittest/badge.svg?branch=main)](https://github.com/livingbio/gpt-fn/actions?query=workflow%3Apython-unittest++branch%3Amain++)
[![Coverage Status](https://coveralls.io/repos/github/livingbio/gpt-fn/badge.svg?branch=main)](https://coveralls.io/github/livingbio/gpt-fn?branch=main)
[![pypi](https://img.shields.io/pypi/v/gpt-fn.svg)](https://pypi.python.org/pypi/gpt-fn)
[![downloads](https://pepy.tech/badge/gpt-fn/month)](https://pepy.tech/project/gpt-fn)
[![versions](https://img.shields.io/pypi/pyversions/gpt-fn.svg)](https://github.com/livingbio/gpt-fn)
[![license](https://img.shields.io/github/license/livingbio/gpt-fn.svg)](https://github.com/livingbio/gpt-fn/blob/main/LICENSE)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


GPT-Fn is a lightweight utility library designed to seamlessly integrate AI capabilities into your software applications. Our focus is on providing essential utilities that make it easy to incorporate artificial intelligence into your codebase without unnecessary complexities.

## Features

- **Function-like API**: With GPT-Fn, you can utilize AI capabilities in your code just like any other function. No need to learn complex AI frameworks or APIs; simply call our functions and harness the power of AI effortlessly.

- **AI Integration**: GPT-Fn seamlessly integrates state-of-the-art AI models, allowing you to perform tasks such as natural language processing, image recognition, sentiment analysis, and much more.

- **Flexible Configuration**: We provide a range of configurable options to fine-tune the behavior of AI functions according to your specific requirements. Customize the models, parameters, and output formats to suit your application's needs.

- **Well Tested**: GPT-Fn comes with a comprehensive suite of test cases, ensuring the reliability and stability of the library. We strive to provide a robust solution that you can trust in your production environments.

- **Open-Source**: GPT-Fn is an open-source project, enabling collaboration and contribution from the developer community. Feel free to explore the source code, suggest improvements, and contribute to making GPT-Fn even more powerful.

## Installation

You can install GPT-Fn using pip, the Python package manager:

```bash
pip install gpt-fn
```

## Getting Started

To start using GPT-Fn in your project, import the library and call the desired function:

```python
from gpt_fn.completion import chat_completion

generated_text = chat_completion(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ],
)

print(generated_text)
```

In the example above, we use the `chat_completion` function to generate response by AI. `chat_completion` also raises error on incomplete responses. The implementation of `chat_completion` makes the most common use case easy. You can explore other available functions in the GPT-Fn documentation/[tests](src/gpt_fn/tests/) for a wide range of AI tasks.

## Contributing

We welcome contributions from the developer community to help improve GPT-Fn. If you encounter any issues, have ideas for new features, or would like to contribute code, please check out our [contribution guidelines](CONTRIBUTING.md). We appreciate your support!

## License

GPT-Fn is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it in your projects. Refer to the license file for more information.

## Acknowledgements

We would like to thank the open-source community for their valuable contributions and the creators of the underlying AI models that power GPT-Fn.

## Contact

If you have any questions, suggestions, or feedback, please don't hesitate to opena an issue.
