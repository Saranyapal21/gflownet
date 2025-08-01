.. _write documentation:

##########################
How to write documentation
##########################

In this document you will learn how to write good, informative, pretty and actionable documentation.

It's not hard!

.. tip::
   **New to documentation?** Start with the :ref:`quick start guide` below, then dive into the details.

.. _quick start guide:

Quick Start for Documentation Contributors
------------------------------------------

**Want to contribute but not sure where to start?** Here are the most common documentation tasks:

1. **Fix a typo or improve existing text**: Just edit the ``.rst`` file and submit a PR
2. **Add a new tutorial**: Create a new ``.rst`` file in ``docs/source/tutorials/``
3. **Improve code documentation**: Edit the docstrings in the Python files
4. **Test your changes locally**: See :ref:`building documentation locally` below

.. _building documentation locally:

Building Documentation Locally
-------------------------------

To write documentation or test your changes, you'll want to **build the documentation locally** on your computer and open the generated HTML files in your browser.

**Install documentation dependencies:**

.. code-block:: bash

    pip install -r docs/requirements-docs.txt

**Build the documentation:**

.. code-block:: bash

    cd docs/
    make html

**View the documentation:**

Open the generated ``docs/_build/html/index.html`` file in your browser:

.. code-block:: bash

    # On most systems:
    open docs/_build/html/index.html
    
    # Or use a simple HTTP server:
    cd docs/_build/html
    python -m http.server 8000
    # Then visit http://localhost:8000

.. note::

    The repository is currently setup to automatically build the documentation on every push to specific branches, including the ``main`` branch. Ask the maintainers if you want your branch to be automatically built too.

Overview
--------

There are two major types of documentation:

1. **docstrings**: your code's docstrings will be automatically parsed by the documentation sofware (`Sphinx <https://www.sphinx-doc.org>`_, more in :ref:`about shpinx`).
2. **Manual** documentation such as this document. This can be for instance a detailed installation procedure, a tutorial, a FAQ, a contributor's guide etc. you name it!

**Both** are written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ (``.rst``) format.

In this tutorial, we'll go over the basics of ``.rst`` and Sphinx, and then we'll look at some of the cool features that are available. To learn about docstrings specifically (what the conventions are, how to write good docstrings etc.), check out the :doc:`next section </contributors/write-docstrings>`.

Some of the great features of using Sphinx is to be able to automatically generate documentation from your code's docstrings, and to be able to link to other parts of the documentation.

For instance: :meth:`~gflownet.gflownet.GFlowNetAgent.trajectorybalance_loss` or to an external function :func:`torch.cuda.synchronize()`.

.. _learn by example:

Learn by example
^^^^^^^^^^^^^^^^

The next section will introduce many of the cool features of ``.rst`` + Sphinx + plugins.

Click on "*Code for the example*" to look at the ``.rst`` code that generated what you are reading.

.. tab-set::

    .. tab-item:: Full-fledged ``.rst`` example

        .. include:: example.rst

    .. tab-item:: Code for the example

        .. literalinclude:: example.rst
            :language: rst

.. note::

    The above tabulation with "Full-fledged ``.rst`` example" and Code for the example was generated using the following code:

    .. code-block:: rst

        .. tab-set::

            .. tab-item:: Full-fledged ``.rst`` example

                .. include:: example.rst

            .. tab-item:: Code for the example

                .. literalinclude:: example.rst
                    :language: rst

FAQ
---

.. dropdown:: How do I create new manual documentation files.

    - Create a new ``.rst`` file in the ``docs/`` folder
    - List it in ``docs/index.rst`` file under the ``.. toctree::`` directive
    - **Or** create a subfolder in ``docs/`` with an ``index.rst`` file.
        - This is useful for grouping documentation files together.
        - ``docs/{your_subfolder}/index.rst`` should contain a ``.. toctree::`` directive listing the files in the subfolder.
        - It should also be listed in the ``docs/index.rst`` under the ``.. toctree::`` directive to appear on the left handside of the documentation.

    You can look at the |contributors|_ folder for an example.

.. dropdown:: How do I document a sub-package like :py:mod:`gflownet.proxy.crystals`?

    Just add a docstring at the top of the ``__init__.py`` file of the sub-package:

    .. code-block:: python

        """
        This is the docstring of the sub-package.

        It can contain any kind of ``.rst`` syntax.

        And refer to its members: :meth:`~gflownet.proxy.crystals.crystal.Stage`

        .. note::

            This is a note admonition.

        """

    You can similarly document a **module** by adding a docstring at the top of the file

.. dropdown:: How do I document a module variable?

    Add a docstring **below** the variable to document like

    .. code-block:: python

        MY_VARIABLE = 42
        """
        This is the docstring of the variable.

        Again, It can contain any kind of ``.rst`` syntax.
        """

.. dropdown:: How do I document a class?

    Currently, ``autoapi`` is setup to consider the documention of a class to be the same as the documentation for the ``__init__`` method of the class.

    This can be modified by changing the ``autoapi_python_class_content = "init"`` configuration variable in ``docs/conf.py``. See `AutoAPI <https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_python_class_content>`_ for more details.

.. dropdown:: (:octicon:`alert` advanced) How do I modify the main API Reference page?

    The main page (that lists sub-modules and packages etc.) is generated by ``autoapi``, using a template file ``docs/_templates/autoapi/index.rst``.

    Modify this file to change the main API Reference page.

    .. important::

        You will notice ``{% ... %}`` blocks. These are `Jinja2 <https://jinja.palletsprojects.com/en/3.0.x/>`_ blocks, a templating language. You can modify them, but be careful not to break the template.

.. dropdown:: (:octicon:`alert` advanced) How do I modify the structure of the class / method / package / module etc. pages?

    The structure of the pages is defined by the ``autoapi`` template files in ``docs/_templates/autoapi/``.

    Modify these files to change the structure of the pages.

    .. important::

        You will notice ``{% ... %}`` blocks. These are `Jinja2 <https://jinja.palletsprojects.com/en/3.0.x/>`_ blocks, a templating language. You can modify them, but be careful not to break the template.


.. dropdown:: Where is the documentation for those advanced features? (tabs, dropdowns etc.)

    - `Sphinx-Design <https://sphinx-design.readthedocs.io/en/furo-theme/>`_ contains many components you can re-use
    - We use the `Furo <https://pradyunsg.me/furo/reference/admonitions/>`_ theme, you'll find the list of available *admonitions* there

.. dropdown:: What plugins are used to make the documentation?

    - `Todo <https://www.sphinx-doc.org/en/master/usage/extensions/todo.html>`_ enables the ``.. todo::`` admonition
    - `Intersphinx mapping <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ enables linking to external documentation like in the ``torch.cuda.synchronize()`` example above
    - `AutoAPI <https://autoapi.readthedocs.io/>`_ enables the automatic generation of documentation from docstrings & package structure
    - `Sphinx Math Dollar <https://www.sympy.org/sphinx-math-dollar/>`_ enables the ``$...$`` math syntax
    - `Sphinx autodoc type ints <https://github.com/tox-dev/sphinx-autodoc-typehints>`_ enables more fine-grained control on how types are displayed in the docs
    - `MyST <https://myst-parser.readthedocs.io/en/latest/intro.html>`_ enables the parsing of enhanced Markdown syntax in the ``.rst`` documentation.
    - `Hover X Ref <https://sphinx-hoverxref.readthedocs.io/en/latest/index.html>`_ Enables tooltips to display contents on the hover of links
    - `Napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ enables the parsing of Google-style docstrings

.. _about shpinx:

About Sphinx
------------

`Sphinx <https://www.sphinx-doc.org>`_ is a documentation generator. It works by parsing ``.rst`` files and generating HTML files from them.

It is configured by the ``docs/conf.py`` file.

To simplify the generation of documentation, we use the `AutoAPI <https://autoapi.readthedocs.io/>`_ plugin, which automatically generates documentation from the package's structure and the docstrings of the code.

AutoAPI reads the code, and generates ``.rst`` files in the ``docs/_autoapi`` folder. These files are then parsed by Sphinx to generate the documentation but to keep the documentation clean, we don't want to commit these files to the repository so ``autoapi`` is configured to delete those ``.rst`` files after generating the documentation.

By default, the generated documentation will be put in the ``API Reference`` section of the overall documentation.


..
    This is a comment.

    LINKS SECTION ⬇️

.. |contributors| replace::  ``docs/contributors/``
.. _contributors: https://github.com/alexhernandezgarcia/gflownet/tree/master/docs/contributors
