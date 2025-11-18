Installation
============

This guide will help you install lostml and set up your development environment.

Prerequisites
-------------

lostml requires:

- **Python 3.7+** - lostml is compatible with Python 3.7 and above
- **NumPy** - For numerical computations (automatically installed)

Installation Methods
--------------------

Method 1: Install from Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/lostml.git
   cd lostml
   pip install -e .

This installs lostml in "editable" mode, meaning any changes you make to the source code will be immediately available without reinstalling.

Method 2: Install Dependencies Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to install dependencies manually:

.. code-block:: bash

   pip install numpy

Then you can import lostml directly if it's in your Python path.

Verifying Installation
----------------------

To verify that lostml is installed correctly, try importing it:

.. code-block:: python

   import lostml
   from lostml import LinearRegression
   from lostml.neighbors import KNN
   
   print("lostml installed successfully!")

If you see no errors, you're all set!

Development Setup
-----------------

For contributing to lostml or running tests:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/yourusername/lostml.git
      cd lostml

2. **Install in editable mode:**

   .. code-block:: bash

      pip install -e .

3. **Install development dependencies:**

   .. code-block:: bash

      pip install pytest

4. **Run tests:**

   .. code-block:: bash

      pytest

Troubleshooting
---------------

**Import Error: No module named 'lostml'**
   Make sure you've installed the package using ``pip install -e .`` from the repository root.

**NumPy not found**
   Install NumPy: ``pip install numpy``

**Permission errors on installation**
   Use ``pip install --user -e .`` to install in user space, or use a virtual environment.

Virtual Environment (Recommended)
----------------------------------

It's recommended to use a virtual environment to avoid conflicts:

.. code-block:: bash

   # Create virtual environment
   python -m venv venv

   # Activate it
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate

   # Install lostml
   pip install -e .
