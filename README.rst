=====
mimpy
=====
Mimetic Finite Differences in Python


The best way to get the code right now is to clone the git repo and run the setup utility:


    .. code-block:: bash
    
        $ git clone https://github.com/ohinai/pas.git
        $ python setup.py install 

An simple example of using the code can be found in (examples/hexmesh/example1/hexmesh_example_1.py).
Run:

    .. code-block:: bash
    
        $ python hexmesh_example_1.py 

If all goes well, you should get a file named (hexmesh_example_1.vtk). Open the file using 
Paraview and plot "MFDPressure." You should see something like this:

.. image:: hexmesh_solution.png







