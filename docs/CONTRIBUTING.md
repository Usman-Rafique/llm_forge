# Contributing to LLM Forge

We welcome contributions to the LLM Forge project! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/llm_forge.git
   cd llm_forge
   ```
3. Set up your development environment as described in [docs/setup.md](setup.md).

## Making Changes

1. Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, following the coding style of the project.
3. Write or update tests as necessary.
4. Run the tests to ensure they pass:
   ```
   python -m unittest discover tests
   ```
5. Commit your changes with a clear and descriptive commit message.

## Submitting a Pull Request

1. Push your changes to your fork on GitHub:
   ```
   git push origin feature/your-feature-name
   ```
2. Open a pull request against the main repository's `main` branch.
3. Provide a clear description of the changes in your pull request.
4. Wait for a maintainer to review your pull request. Address any feedback or comments they provide.

## Coding Guidelines

- Follow Google's Python Style Guide using yapf for formatting.
- Write clear, self-documenting code with appropriate comments where necessary.
- Keep functions and methods focused and concise.

## Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the GitHub issue tracker.
2. If not, create a new issue, providing as much detail as possible, including:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Any error messages or logs

## Questions or Need Help?

If you have any questions or need help with contributing, please feel free to open an issue or reach out to the maintainers.

Thank you for contributing to LLM Forge!
