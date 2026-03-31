#------------------------------------------------------------------------------
#  dtOO < design tool Object-Oriented >
#    
#    Copyright (C) 2024 A. Tismer.
#------------------------------------------------------------------------------
#License
#    This file is part of dtOO.
#
#    dtOO is distributed in the hope that it will be useful, but WITHOUT
#    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#    FITNESS FOR A PARTICULAR PURPOSE.  See the LICENSE.txt file in the
#    dtOO root directory for more details.
#
#    You should have received a copy of the License along with dtOO.
#
#------------------------------------------------------------------------------

#
# Import packages
#
import numpy as np
import dtOOPythonSWIG as dtOO
import foamlib as fl
import sys
import subprocess
import time
from utils import * 
import os
import shutil
import traceback
logger = import_logger()
#
# Define the values that will be stored during the optimization; here only the
# name and the initial definition for each stored value is declared; the
# arrays objective and fitness are created by default
#
import pyDtOO as pd

class hydFoil:
  """Create, mesh, simulate and evaluate a hydrofoil.

  This class holds all functions to create a hydrofoil with an inlet angle,
  outlet angle and a blade thickness. :numref:`hydfoil` shows a sketch of the 
  hydrofoil.

  .. _hydfoil:
  .. figure:: img/hydfoil.png
     :width: 600
     :align: center

     Hydrofoil's sketch including mean line (solid thick black line) and final
     shape (solid thin black line); B-Splines, that are used for constructing
     the meanline and final shape, are shown as solid and dashed gray thin
     lines; velocity triangles at inlet and outlet of the hydrofoil are 
     colored in magenta; the DOFs, namely :math:`\\alpha_1`,
     :math:`\\alpha_2`, and :math:`t_{mid}`, are shown and labeled in black

  The simulation of the hydrofoil is performed in the relational frame of
  reference. It means that all velocities in the CFD simulation correspond to
  :math:`w`. Therefore, when evaluating hydrofoil's efficiency and head, the
  transformation

  .. math::

    c = w + u = w + 2 \\pi n R \\approx w + 18.84 \\frac{m}{s}
  
  is necessary to get the absolute velocity :math:`c`. The hydrofoil is 
  designed for a design head of :math:`0.8 m`. The fitness function is 
  calculated based on head deviation and efficiency.

  Parameters
  ----------
  alpha_1: float
    Inlet angle.
  alpha_2: float
    Outlet angle.
  t_mid: float
    Blade's thickness.
  """
  
  import sys
  
  H_          = 0.2
  """float: Height of the mesh in :math:`m`."""
  R_          = 2.0
  """float: Radius of the blade cut in :math:`m` where this hydrofoil is 
            located."""
  nB_         = 4
  """float: Number of blades in which this hydrofoil is located."""
  L_          = 4.0
  """float: Length of the mesh in :math:`m`."""
  n_          = 90.0
  """float: Rotational speed in :math:`min^{-1}`."""
  c_mi_       = 5.77
  """float: Absolute velocity at inlet in :math:`\\frac{m}{s}`"""

  def __init__(
    self, 
    alpha_1 = 100.0,  
    alpha_2 = 130, 
    t_mid = 0.1,
    state = None
  ):
    ### change working directory to local ssd ###
    self.wd = os.environ["TMPDIR"]
    os.chdir(self.wd)
    """str: State label."""
    self.state_ = state
    dtOO.logMe.initLog('build.'+self.state_+'.log')
    self.history_ = {} 
    working_dir =f'{{\"name\": \"workingDirectory\", \"value\": \"{self.wd}\" }}'
    jsonPrimitive_string = (
        '{'
          '"option" : ['
            '{"name" : "reparamOnFace_precision", "value" : "1.e-06"},'
            '{"name" : "reparamInVolume_precision","value" : "1.e-06"},'
            '{"name" : "reparam_internalRestarts", "value" : "10"},'
            '{"name" : "reparam_restarts", "value" : "10"},'
            '{"name" : "reparam_restartIncreasePrecision", "value" : "10."},'
            '{'
              '"name" : "reparam_internalRestartDecreasePrecision",'
              ' "value" : "0.9"'
            '},'
            '{"name" : "invY_precision", "value" : "1.e-04"},'
            '{"name" : "xyz_resolution", "value" : "1.e-08"},'
            '{"name" : "XYZ_resolution", "value" : "1.e-07"},'
            '{"name" : "uvw_resolution", "value" : "1.e-04"},'
            '{"name" : "root_printLevel", "value" : "0"},'		
            '{"name" : "root_maxIterations", "value" : "1000"},'
            '{"name" : "root_maxFunctionCalls", "value" : "1000000"},'
            '{"name" : "logLevel", "value" : "99"},' + working_dir + ']'
        '}')
    #logger.info(jsonPrimitive_string)
    dtOO.staticPropertiesHandler.getInstance().jInit(
      dtOO.jsonPrimitive(jsonPrimitive_string))
      #  '{'
      #    '"option" : ['
      #      '{"name" : "reparamOnFace_precision", "value" : "1.e-06"},'
      #      '{"name" : "reparamInVolume_precision","value" : "1.e-06"},'
      #      '{"name" : "reparam_internalRestarts", "value" : "10"},'
      #      '{"name" : "reparam_restarts", "value" : "10"},'
      #      '{"name" : "reparam_restartIncreasePrecision", "value" : "10."},'
      #      '{'
      #        '"name" : "reparam_internalRestartDecreasePrecision",'
      #        ' "value" : "0.9"'
      #      '},'
      #      '{"name" : "invY_precision", "value" : "1.e-04"},'
      #      '{"name" : "xyz_resolution", "value" : "1.e-08"},'
      #      '{"name" : "XYZ_resolution", "value" : "1.e-07"},'
      #      '{"name" : "uvw_resolution", "value" : "1.e-04"},'
      #      '{"name" : "root_printLevel", "value" : "0"},'		
      #      '{"name" : "root_maxIterations", "value" : "1000"},'
      #      '{"name" : "root_maxFunctionCalls", "value" : "1000000"},'
      #      '{"name" : "logLevel", "value" : "99"}'
      #  ']'
      #  '}'
     #)
    #)
    
    self.container = dtOO.dtBundle()
    """dtOOPythonSWIG.dtBundle: Bundle object."""
    
    self.bC = self.container.cptr_bC()
    """dtOOPythonSWIG.baseContainer: base container."""
    self.cV = self.container.cptr_cV()
    """dtOOPythonSWIG.labeledVectorHandlingConstValue: Container object 
    of dtOOPythonSWIG.constValue."""
    self.aF = self.container.cptr_aF() 
    """dtOOPythonSWIG.labeledVectorHandlingAnalyticFunction: Container object 
    of dtOOPythonSWIG.analyticFunction."""
    self.aG = self.container.cptr_aG() 
    """dtOOPythonSWIG.labeledVectorHandlingAnalyticGeometry: Container object 
    of dtOOPythonSWIG.analyticGeometry."""
    self.bV = self.container.cptr_bV() 
    """dtOOPythonSWIG.labeledVectorHandlingBoundedVolume: Container object 
    of dtOOPythonSWIG.boundedVolume."""
    self.dC = self.container.cptr_dC() 
    """dtOOPythonSWIG.labeledVectorHandlingDtCase: Container object 
    of dtOOPythonSWIG.dtCase."""
    self.dP = self.container.cptr_dP() 
    """dtOOPythonSWIG.labeledVectorHandlingDtPlugin: Container object 
    of dtOOPythonSWIG.dtPlugin."""

    #
    # Create and initialize constValues for DOFs; the objects are cloned and
    # appended to the container; it is necessary to create a clone, otherwise
    # the instance is destructed at the end of this function; it is also 
    # possible to use the thisown flag that is implemented via SWIG, see
    # documentation of SWIG 
    # https://www.swig.org/Doc4.1/SWIGDocumentation.html#Python_nn28
    #
    self.cV.set(
      dtOO.sliderFloatParam("alpha_1", alpha_1, 130.0, 170.0).clone()
    )
    self.cV.set(
      dtOO.sliderFloatParam("alpha_2", alpha_2, 130.0, 170.0).clone()
    )
    self.cV.set(
      dtOO.sliderFloatParam("t_mid", t_mid, 0.01, 0.70).clone()
    )

    #
    # Add a lVHOstateHandler object to create state labels; clearing is 
    # necessary to prevent memory corruption; thisown is necessary to make
    # sure that object is not destructed at the end of this function
    #
    dtOO.lVHOstateHandler.clear()
    dtOO.lVHOstateHandler( dtOO.jsonPrimitive(), self.cV ).thisown = False

    #
    # Create a state using the lVHOstateHandler; this forces also the creation
    # of a json file <statename>.json written to disk
    #
    dtOO.lVHOstateHandler().makeState(self.state_)


  def Geometry(self):
    """Create hyrdofoil's geometry.
  
    The main objects of baseContainer, analyticFunction, and analyticGeometry
    are created. Objects that are necessary or interesting are appended to the
    :attr:`hydFoilOpt.build.hydFoil.bV`, :attr:`hydFoilOpt.build.hydFoil.aF`, 
    and :attr:`hydFoilOpt.build.hydFoil.aG`.
    """
   
    #
    # Calculate the width of the channel; it is given by the fraction of the 
    # unwounded length 
    # 
    twoPiRByNB = 2.0*np.pi*hydFoil.R_/hydFoil.nB_
    
    #
    # Create 4 points to define the periodic surface located on the left side;
    # points are objects of dtPoint3; coordinates are accessible using python's
    # index notation ([]-operator)
    #
    P1 = dtOO.dtPoint3(-0.5*hydFoil.H_, -0.5*twoPiRByNB, 0)
    P2 = dtOO.dtPoint3(P1[0], P1[1], hydFoil.L_)
    P3 = dtOO.dtPoint3(P1[0]+hydFoil.H_, P1[1], P1[2])
    P4 = dtOO.dtPoint3(P2[0]+hydFoil.H_, P2[1], P2[2])
   
    #
    # Create B-Spline surface by skinning two B-Spline lines; the lines are
    # straight connections, because order is not specified; therefore default
    # value of 1 is used
    #
    perio = dtOO.analyticSurface(
      dtOO.bSplineSurface_skinConstructOCC(
        dtOO.bSplineCurve_pointConstructOCC(P1, P2).result(),
        dtOO.bSplineCurve_pointConstructOCC(P3, P4).result()
      ).result()
    )
   
    #
    # Create channel volume by translating B-Spline surface in y-direction
    #
    channel = dtOO.translatingMap2dTo3d(
     dtOO.dtVector3(0.0, twoPiRByNB, 0.0), perio
    )
    
    #
    # Define label for the channel volume
    #
    channel.setLabel("xyz_channel")
    
    #
    # Append a clone of the channel volume to the analyticGeometry container;
    # as an alternative without cloning, the thisown flag of channel volume
    # must be set to 0
    #
    self.aG.set(channel.clone())
   
    #
    # Create a conformal mapping to map between parameter and physical 
    # coordinates; the transformer is initialized with a JSON object that
    # defines the label, the geometry "to-map-to", the number of points in
    # v-direction, and the number of points in w-direction; it is necessary
    # to give self.aG as input argument, because the transformer clones the
    # geometry "to-map-to" and keeps an instance as an attribute; therefore, a
    # geometry change of "xyz_channel" after the next statement is useless
    # for the transformer; the internal "to-map-to" geometry is not update;
    # store a clone in the analyticGeometry container, too
    #
    cMap = dtOO.uVw_deltaMs()
    cMap.jInit(
      dtOO.jsonPrimitive()
        .appendStr("label", "cMap")
        .appendAnalyticGeometry("_tM2d", self.aG.get("xyz_channel"))
        .appendInt("_nV", 31)
        .appendInt("_nW", 11),
      None, None, None, self.aG
    )
    self.bC.ptrTransformerContainer().add( cMap.clone() )

    #
    # Extract instances of constValue by their labels ("alpha_1" and 
    # "alpha_2"); a call to the ()-operator on a constValue object, returns
    # its internal value
    #
    alpha_1 = self.cV["alpha_1"]()
    alpha_2 = self.cV["alpha_2"]()

    #
    # Define additional parameter variables and assign values; theoretically,
    # those variables could also be modifiable, but the example is kept as
    # small as possible; therefore those are fix
    #
    ratio = 0.5
    deltaM = 0.30
    offM   = 0.75
    bladeLength = 0.70
   
    #
    # Import predefined builder to create a mean plane based on alphaOne,
    # alphaTwo, ratioX, deltaY, offX, offY and targetLength, see corresponding
    # documentation for details; the DOFs are prescribed as functions; in this
    # case, the functions are a linear curve between two points; the builder 
    # gets and returns a dtBundle object; the generated mean plane is appended 
    # to the input dtBundle object; all created objects are labeled with the 
    # given tag; in the case at hand, it is simply "meanplane"
    #
    from dtOOPythonApp.builder import (
      analyticSurface_threePointMeanplaneFromRatio,
      scaOneD_scaCurve2dOneDPointConstruct
    )
    self.container = analyticSurface_threePointMeanplaneFromRatio(
      "meanplane",
      spanwiseCuts = [
        0.00,  
        1.00,
      ],
      alphaOne = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, (np.pi/180.) * alpha_1),  
          dtOO.dtPoint2(1.00, (np.pi/180.) * alpha_1),
        ],
        1
      )(),
      alphaTwo = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, (np.pi/180.) * alpha_2),  
          dtOO.dtPoint2(1.00, (np.pi/180.) * alpha_2),
        ],
        1
      )(),
      ratioX = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, ratio),
          dtOO.dtPoint2(1.00, ratio),  
        ],
        1
      )(),
      deltaY = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, deltaM),
          dtOO.dtPoint2(1.00, deltaM),  
        ],
        1
      )(),
      offX = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, 0.5*twoPiRByNB),  
          dtOO.dtPoint2(1.00, 0.5*twoPiRByNB)
        ],
        1
      )(),
      offY = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, offM),  
          dtOO.dtPoint2(1.00, offM),
        ],
        1
      )(),
      targetLength = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, bladeLength),  
          dtOO.dtPoint2(1.00, bladeLength),
        ],
        1
      )(),
      targetLengthTolerance = 0.01,
      originOnLengthPercent = 0.5
    ).buildExtract( self.container )

    #
    # At first, extract the DOF "t_mid" for the thickness distribution out of
    # the container by its label; then, import and apply a predefined builder
    # for a B-Spline thickness distribution defined by 5 control points
    #
    t_mid = self.cV["t_mid"]()
    from dtOOPythonApp.builder import (
      vec3dSurfaceTwoD_fivePointsBSplineThicknessDistribution
    )
    self.container = vec3dSurfaceTwoD_fivePointsBSplineThicknessDistribution(
      "thicknessDistribution",
      spanwiseCuts = [
        0.00,  
        1.00,
      ],
      tLe = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, 0.05),  
          dtOO.dtPoint2(1.00, 0.05),
        ],
        1
      )(),
      uLe = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, 0.00),  
          dtOO.dtPoint2(1.00, 0.00),
        ],
        1
      )(),
      tMid = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, t_mid),  
          dtOO.dtPoint2(1.00, t_mid),
        ],
        1
      )(),
      uMid = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, 0.50),  
          dtOO.dtPoint2(1.00, 0.50),
        ],
        1
      )(),
      tTe = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, 0.01),  
          dtOO.dtPoint2(1.00, 0.01),
        ],
        1
      )(),
      uTe = scaOneD_scaCurve2dOneDPointConstruct(
        [
          dtOO.dtPoint2(0.00, 0.80),  
          dtOO.dtPoint2(1.00, 0.80),
        ],
        1
      )()
    ).buildExtract( self.container )
   
    #
    # Define a transformer to combine the mean plane and the thickness
    # distribution; the transformer adds the thickness to the mean plane in
    # perpendicular direction; a number of predefined points in u- and 
    # v-direction, respectively, "_nU" and "_nV", are transformed to form
    # the final hydrofoil's shape; the points are resplined and, therefore,
    # connected by a B-Spline of order 3 ("_order"); as already explained the
    # transformer clones the "thicknessDistribution" to keep an internal
    # instance and, in that sense, it is necessary to have self.aF as an 
    # inlet argument
    #
    dAdd = dtOO.discreteAddNormal()
    dAdd.jInit(
      dtOO.jsonPrimitive(
        '{"option" : [{"name" : "debug", "value" : "false"}]}'
      )
        .appendAnalyticFunction("_tt", self.aF["thicknessDistribution"])
        .appendInt("_nU", 61)
        .appendInt("_nV", 41)
        .appendInt("_order", 3)
        .appendDtVector3("_nf", dtOO.dtVector3(0,0,1)),
      None, None, self.aF, None
    )

    #
    # Apply the defined transformer to the "meanplane" function; set a label
    # and append the object to the analyticFunction container
    #
    theAF = dAdd.applyAnalyticFunction( self.aF["meanplane"] )
    theAF.setLabel("blade")
    self.aF.set( theAF.clone() )

  def GeometryMesh(self):
    """Create additional hydrofoil's geometry for meshing.

    Create additional objects that are necessary for creating the mesh. These
    are mainly object's of the mesh block that is created around the blade.
    This block is meshed as a structured block.
    """
    #logger.info(f"Dir in Geometry_Mesh: {os.getcwd()}.")
    #
    # Create an analyticFunction that maps (u,v)->(t,u,v) with u, v, and t or
    # first parameter coordinate, second parameter coordinate, and thickness,
    # respectively; the thickness is fixed to 0.1; set the label and define
    # the bounds of first and second function argument
    #
    fRef = dtOO.vec3dMuParserTwoD("0.10, uu, vv", "uu", "vv")
    fRef.setLabel("thicknessMeshBlock")
    for i in range(2):
      fRef.setMin(i, +0.0)
      fRef.setMax(i, +1.0)
   
    #
    # Create an temporary analyticFunction container to store the 
    # analyticFunction; this is necessary to initialize the following
    # transformer; as an alternative the analyticFunction can also be appended
    # to :attr:`hydFoilOpt.build.hydFoil.aF`
    #
    tmpAF = dtOO.labeledVectorHandlingAnalyticFunction()
    tmpAF.set( fRef.clone() )
   
    #
    # Define the transformer to create the outer boundary of the mesh
    # blocks; it is necessary to have a second transformer, because they
    # operate on different "_tt" attributes
    #
    dAdd = dtOO.discreteAddNormal()
    dAdd.jInit(
      dtOO.jsonPrimitive(
        '{"option" : [{"name" : "debug", "value" : "false"}]}'
        ) \
        .appendAnalyticFunction("_tt", tmpAF["thicknessMeshBlock"]) \
        .appendInt("_nU", 61) \
        .appendInt("_nV", 41) \
        .appendInt("_order", 3) \
        .appendDtVector3("_nf", dtOO.dtVector3(0,0,1)),
      None, None, tmpAF, None
    )
   
    #
    # Apply transformer to blade geometry and append it to the corresponding
    # container
    #
    theAF = dAdd.applyAnalyticFunction( self.aF["blade"] )
    theAF.setLabel("meshBlock")
    self.aF.set( theAF.clone() )
    
    #
    # Import predefined builder to create the mesh blocks; they are created by
    # skinning between the "blade" and "meshBlock" geometry in combination
    # with splitting the final three dimensional region; the resulting
    # geometries are appended to the dtBundle object
    #
    from dtOOPythonApp.builder import vec3dThreeD_skinAndSplit
    self.container = vec3dThreeD_skinAndSplit(
      label = "meshBlock",
      aFOne = self.aF["blade"],
      aFTwo = self.aF["meshBlock"],
      splitDim = 0,
      splits = [
        [0.00, 0.10],
        [0.10, 0.30],
        [0.30, 0.45],
        [0.45, 0.55],
        [0.55, 0.70],
        [0.70, 0.90],
        [0.90, 1.00],
      ]
    ).buildExtract(self.container)
  
    #
    # Get a reference of the conformal mapping transformer
    #
    cMap = self.bC.ptrTransformerContainer()["cMap"]

    #
    # Perform the conformal mapping of the two dimensional B-Spline geometries
    # between parameter space (u,v) and physical space (x,y,z); the result is
    # a composition of multiple functions; append for each geometry a shape to
    # the analyticGeometry container
    #
    for ii in ["meanplane", "blade", "meshBlock",]:
      theAG = dtOO.vec3dTwoDInMap3dTo3d(
        dtOO.vec3dTwoD.MustConstDownCast(
          cMap.applyAnalyticFunction( self.aF[ii].clone() )
        ),
        dtOO.map3dTo3d.ConstDownCast( self.aG["xyz_channel"] )
      )
      theAG.setLabel("xyz_"+ii)
      self.aG.set( theAG.clone() )
   
    #
    # Perform the conformal mapping of the three dimensional B-Spline 
    # geometries between parameter space (u,v,w) and (x,y,z)
    #
    for iNum in self.aF.getIndices("meshBlock_*"):
      ii = self.aF.getLabel( iNum )
      theAG = dtOO.vec3dThreeDInMap3dTo3d(
        dtOO.vec3dThreeD.MustConstDownCast(
          cMap.applyAnalyticFunction( self.aF[ii].clone() )
        ),
        dtOO.map3dTo3d.ConstDownCast( self.aG["xyz_channel"] )
      )
      theAG.setLabel("xyz_"+ii)
      self.aG.set( theAG.clone() )

  def Mesh(self):
    """Create hydrofoil's mesh.
    Create the topology for the hydrofoil's mesh. Additionally, define number
    of elements for edges and surfaces, define minimum and maximum element
    lengths for unstructured meshing algorithms, and label mesh parts.
    """
    #
    # Extract coupling surfaces between mesh blocks and outer region; these
    # surfaces are necessary to define the outer region; in this simple test
    # case the surfaces of interest are clear, because the blocks are ordered;
    # store the blocks and surfaces in two lists "blocks" and "couplingFaces"
    #
    blocks = []
    for iNum in self.aG.getIndices("xyz_meshBlock_*"):
      blocks.append( self.aG[ self.aG.getLabel( iNum ) ] )
    
    couplingFaces = []
    couplingFaces.append( 
      dtOO.map3dTo3d.MustDownCast( blocks[0] ).segmentConstUPercent( 0.0 )
    )
    for block in blocks:
      couplingFaces.append( 
        dtOO.map3dTo3d.MustDownCast( block ).segmentConstWPercent( 1.0 )
      )
    couplingFaces.append( 
      dtOO.map3dTo3d.MustDownCast( blocks[-1] ).segmentConstUPercent( 1.0 )
    )
   
    #
    # Create a boundedVolume object that keeps all geometries, topologies and
    # information for meshing; initialize the boundedVolume with a JSON 
    # structure; gmsh options can be set as shown e.g. for the value
    # "Mesh.CharacteristicLengthMin"; theoretically, the geometries being
    # meshed can also be included in the JSON structure within the 
    # vector of analyticGeometry; in the case at hand, they are added via
    # separate functions
    #
    gBV = dtOO.gmshBoundedVolume()
    gBV.jInit(
      dtOO.jsonPrimitive(
      '{'
        '"label" : "mesh", '
        '"option" : ['
          '{"name" : "[gmsh]General.Terminal", "value" : "1."},'
          '{"name" : "[gmsh]General.Verbosity", "value" : "100."},'
          '{"name" : "[gmsh]General.ExpertMode", "value" : "1."},'
          '{'
            '"name" : "[gmsh]Mesh.LcIntegrationPrecision", '
            '"value" : "1.0E-04"'
          '},'
          '{'
            '"name" : "[gmsh]Mesh.CharacteristicLengthMin", '
            '"value" : "0.1"'
          '},'
          '{'
            '"name" : "[gmsh]Mesh.CharacteristicLengthMax", '
            '"value" : "0.5"'
          '},'
          '{"name" : "[gmsh]Mesh.Algorithm", "value" : "1"},'
          '{'
            '"name" : "[gmsh]Mesh.MeshSizeExtendFromBoundary", '
            '"value" : "1"'
          '},'
          '{"name" : "[gmsh]Mesh.MeshSizeFromPoints", "value" : "1"}'
        '],'
        '"analyticGeometry" : []'
      '}'
      ),
      None, None, None, None, None
    )

    #
    # Append the boundedVolume to its container; prevent destruction by
    # setting thisown to false
    #
    gBV.thisown = False
    self.bV.set( gBV )
   
    #
    # Store the underlying gmsh model in a separate variable to have easy
    # access
    #
    gm = gBV.getModel()
   
    #
    # Add three dimensional outer region to model; the return value is the 
    # internal tag of the region in the gmsh model; 
    #
    tag = gm.addIfRegionToGmshModel( 
      dtOO.map3dTo3d.DownCast( self.aG["xyz_channel"] ) 
    )
   
    #
    # Add a physical tag to the region
    #
    gm.tagPhysical( gm.getRegionByTag(tag), "xyz_channel" )
   
    #
    # Create a list that contains all indices of the mesh blocks; the list is
    # for easy access of the desired geometries
    #
    mbIndices = self.aG.getIndices("xyz_meshBlock_*")
    
    #
    # Add the mesh block geometries to the gmsh model and, additionally, add
    # "NORTH" face of each block to the outer region; those faces are the
    # coupling faces
    #
    for iNum in mbIndices:
      tag = gm.addIfRegionToGmshModel( 
        dtOO.map3dTo3d.DownCast( self.aG[iNum] ) 
      )
      gm.tagPhysical( gm.getRegionByTag(tag), self.aG[iNum].getLabel() )
      gm.getDtGmshRegionByPhysical("xyz_channel").addFace(
        gm.getDtGmshFaceByPhysical(self.aG[iNum].getLabel()+"->NORTH"), 1 
      )

    #
    # Add "WEST" surface of first and "EAST" surface of last mesh block to
    # outer region; both surfaces are also coupling surfaces
    #
    gm.getDtGmshRegionByPhysical("xyz_channel").addFace(
     gm.getDtGmshFaceByPhysical("xyz_meshBlock_0->WEST"), 1 
    )
    gm.getDtGmshRegionByPhysical("xyz_channel").addFace(
      gm.getDtGmshFaceByPhysical(
        "xyz_meshBlock_"+str(np.size(mbIndices)-1)+"->EAST"
      ), 
      1 
    )
   
    #
    # Create an observer to automatically detect internal edges of the outer 
    # region "xyz_channel"; the observer extracts all edges that lie within 
    # the "NORTH" and "SOUTH" face of the region; the extracted edges are then
    # oriented to form an edge loop
    #
    ob = dtOO.bVOAddInternalEdge()
    ob.jInit(
      dtOO.jsonPrimitive('{ "_regionLabel" : "xyz_channel"}'), 
      None, None, None, None, None, gBV
    )

    #
    # Apply observer with the "preUpdate" function; in general, observers can
    # be applied before or after the meshing procedure; theoretically, the 
    # observer can also be appended to the internal observer vector of the
    # bounded volume; if this is the case, all observers within the vector are
    # then automatically applied; in the case at hand, the observer is
    # manually applied
    #
    ob.preUpdate()
   
    #
    # Define number of elements within the mesh blocks; the regions are also
    # defined to be transfinite with a recursive recombination; the latter
    # enables the creation of quadrangles or rather hexahedrons
    #
    for iNum in mbIndices:
      mb = gm.getDtGmshRegionByPhysical( self.aG[iNum].getLabel() )
      mb.meshWNElements(10,1,15)
      mb.meshTransfiniteRecursive()
      mb.meshRecombineRecursive()
    
    #
    # Define number of elements at periodic, inlet, and outlet surface; there
    # is only one element in x-direction, because it is a two dimensional
    # simulation; the surfaces are meshed transfinite and the mesh is then 
    # recombined
    #
    for lab, nU, nV in zip(
      [
        "xyz_channel->EAST", 
        "xyz_channel->WEST", 
        "xyz_channel->FRONT",
        "xyz_channel->BACK",
      ],
      [10,10,1, 1,],
      [1, 1, 10,10,]
    ):
      gm.getDtGmshFaceByPhysical(lab).meshWNElements(10,1)
      gm.getDtGmshFaceByPhysical(lab).meshTransfinite()
      gm.getDtGmshFaceByPhysical(lab).meshRecombine()
    
    #
    # As mentioned above, the case is a two dimensional simulation; therefore,
    # the mesh is periodic on the "NORTH" and "SOUTH" face of the outer 
    # region; the transformation between the nodes is a simple translation in
    # x-direction; it is implemented by using objects of translate
    #
    dtT_hs = dtOO.translate( 
      dtOO.jsonPrimitive().appendDtVector3("_v3", dtOO.dtVector3(hydFoil.H_,0,0))
    )
    
    #
    # The transformer is appended to a temporary baseContainer
    # 
    tmp_bC = dtOO.baseContainer()
    tmp_bC.ptrTransformerContainer().add(dtT_hs)
    
    #
    # Create the observer to handle translational periodicity in gmsh; 
    # customize the observer with a JSON object that sets master and slave 
    # face and, additionally, provide the transformer
    #
    ob = dtOO.bVOSetTranslationalPeriodicity()
    ob.jInit( 
      dtOO.jsonPrimitive(
        '{'
          '"_faceMaster" : "xyz_channel->SOUTH",'
          '"_faceSlave" : "xyz_channel->NORTH"'
        '}'
      ).appendDtTransformer("_dtT", dtT_hs), 
      tmp_bC, None, None, None, None, gBV 
    )

    #
    # Apply transformer by calling "preUpdate" function
    #
    ob.preUpdate()
   
    #
    # Perform the meshing procedure within the "bVOMeshRule" observer; the
    # meshing procedure is customized within the JSON structure; there is a
    # rule for each mesh dimension; within a rule an operator is combined
    # with entities by their labels; the operators are then defined in the
    # "dtMeshOperator" vector
    #
    ob = dtOO.bVOMeshRule()
    ob.jInit(
      dtOO.jsonPrimitive( 
        '{'
          '"option" : ['
            '{"name" : "debug", "value" : "true"}'
          '],'
          '"_rule1D" : ['
            '"dtMeshGEdge(xyz_channel->*->*)",'
            '"dtMeshGEdge(xyz_meshBlock_*->*->*)"'
          '],'
          '"_rule2D" : ['
            '"dtMeshTransfiniteGFace(xyz_meshBlock*->*)",'
            '"dtMeshGFace(xyz_*->*)"'
          '],'
          '"_rule3D" : ['
            '"dtMeshGRegion(xyz_meshBlock*)",'
            '"dtMeshGRegionWithOneLayer(xyz_channel*)"'
          '],'
          '"_only" : [],'
          '"dtMeshOperator" : ['
            '{'
              '"name" : "dtMeshGEdge",'
              '"label" : "dtMeshGEdge"'
            '},'
            '{'
              '"name" : "dtMeshGFace",'
              '"label" : "dtMeshGFace"'
            '},'
            '{'
              '"name" : "dtMeshTransfiniteGFace",'
              '"label" : "dtMeshTransfiniteGFace"'
            '},'
            '{'
              '"name" : "dtMeshGRegion",'
              '"label" : "dtMeshGRegion",'
              '"_minQShapeMetric" : 0.0,' 
              '"_relax" : 0.1,'
              '"_nPyramidOpenSteps" : 10,'
              '"_nSmooths" : 3'
            '},'
            '{'
              '"name" : "dtMeshGRegionWithOneLayer",'
              '"label" : "dtMeshGRegionWithOneLayer",'
              '"_faceMaster" : "xyz_channel->SOUTH",' 
              '"_faceSlave" : "xyz_channel->NORTH"'
            '}'
                        
          ']'
        '}'
      ),
      None, None, None, None, None, gBV
    )

    #
    # Attach the observer to the boundedVolume; this means that it is 
    # automatically executed; it is necessary to set the thisown flag, 
    # otherwise the objects is being destroyed
    #
    gBV.attachBVObserver(ob)
    ob.thisown = False
   
    #
    # Create the mesh within the boundedVolume
    #
    gBV.makeGrid()
   
    #
    # Create an observer that renames internal mesh faces; this cleans up and 
    # creates a clear naming of the faces
    #
    ob = dtOO.bVOFaceToPatchRule()
    ob.jInit(
      dtOO.jsonPrimitive(
        '{'
          '"_patchRule" : ['
            '":xyz_channel->FRONT::INLET:",'
            '":xyz_channel->BACK::OUTLET:",'
            '":xyz_channel->EAST::PERIOA:",'
            '":xyz_channel->WEST::PERIOB:",'
            '":xyz_meshBlock_*->BACK::EMPTYB:",'
            '":xyz_meshBlock_*->FRONT::EMPTYA:",'
            '":xyz_channel->SOUTH::EMPTYA:",'
            '":xyz_channel->NORTH::EMPTYB:",'
            '":xyz_meshBlock_*->SOUTH::BLADE:"'
          '],'
          '"_regRule" : [":*::R:"]'
        '}'
      ), gBV
    )

    #
    # Apply the observer after the mesh is created; this is done by calling
    # "postUpdate"
    #
    ob.postUpdate()
    
    #
    # Create and apply an observer to write the mesh in the gmsh native "msh"
    # format to disk
    #
    ob = dtOO.bVOWriteMSH()
    ob.jInit(
      dtOO.jsonPrimitive( '{"_filename" : "", "_saveAll" : true}' ), gBV 
    )
    ob.postUpdate()
   
    #
    # Create and apply an observer that orients the cell volumes within the
    # mesh; this makes sure to be conform with OpenFoam
    #
    ob = dtOO.bVOOrientCellVolumes()
    ob.jInit(
      dtOO.jsonPrimitive('{ "_positive" : true }'), gBV
    )
    ob.postUpdate()
    
    #
    # Calculate again the width of the channel for defining the periodic
    # surfaces
    #
    twoPiRByNB = 2.0*np.pi*hydFoil.R_/hydFoil.nB_
    
    #
    # Import a predefined builder to setup the OpenFoam case; within the 
    # builder all necessary files are written to disk; it includes an
    # automatic definition of functions for calculating total pressure and
    # discharges on desired patches; additionally all boundary conditions are
    # defined
    #
    from dtOOPythonApp.builder import (
      ofOpenFOAMCase_turboMachine,
      ofOpenFOAMCase_setupWrapper
    )
    self.container = ofOpenFOAMCase_turboMachine(
      label = "of",
      bVs = [
        self.bV["mesh"],
      ],
      dictRule = 
          ofOpenFOAMCase_setupWrapper.controlDict(
            application = "simpleFoam",
            endTime = 100,
            QPatches = ['INLET','OUTLET',],
            PTPatches = ['INLET', 'OUTLET',],
            FPatches = ['BLADE',],
            libs = []
          )
        + ofOpenFOAMCase_setupWrapper.fvSchemes()
        + ofOpenFOAMCase_setupWrapper.fvSolution()
        + ofOpenFOAMCase_setupWrapper.transportModel()
        + ofOpenFOAMCase_setupWrapper.turbulenceProperties(),
        fieldRules = [ 
          ofOpenFOAMCase_setupWrapper.fieldRuleString("U", [0.0,0.0,0.0,]),
          ofOpenFOAMCase_setupWrapper.fieldRuleString("p", [0.0,]),
          ofOpenFOAMCase_setupWrapper.fieldRuleString("k", [0.1,]),
          ofOpenFOAMCase_setupWrapper.fieldRuleString("omega", [0.1,]),
          ofOpenFOAMCase_setupWrapper.fieldRuleString("nut", [0.1,]),
        ],
        setupRules = [
          ofOpenFOAMCase_setupWrapper.emptyRuleString(),
          ofOpenFOAMCase_setupWrapper.inletRuleString(
            "INLET", 
            ["U"], 
            [ [0,-2.0*np.pi*hydFoil.n_/60.*hydFoil.R_,hydFoil.c_mi_], ]
          ), 
          ofOpenFOAMCase_setupWrapper.inletRuleString(
            "INLET", 
            ["p", "k", "omega",], 
            [ [0], [0.10, 0.20], [0.032*hydFoil.R_, 0.1] ]
          ),
          ofOpenFOAMCase_setupWrapper.emptyRuleString(
            "EMPTYA"
          ),
          ofOpenFOAMCase_setupWrapper.emptyRuleString(
            "EMPTYB" 
          ),
          ofOpenFOAMCase_setupWrapper.wallRuleString(
            "BLADE", 
            ["omega", "U", "p", "k", "nut"]
          ),
          ofOpenFOAMCase_setupWrapper.cyclicAmiTranslationalRuleString(
            "PERIOA", "PERIOB", 
            sepVector = dtOO.dtVector3(0,-twoPiRByNB,0)
          ),
          ofOpenFOAMCase_setupWrapper.outletRuleString(
            "OUTLET", 
            ["U", "p", "k", "omega",]
          ),
        ]
    ).buildExtract( self.container )

    #
    # Output to log of the current state label
    #
    logging.info( 
      "Current state is > %s <." % (dtOO.lVHOstateHandler().commonState()) 
    )

    #
    # Run the current state of the dtCase object "of"; this object was created
    # with the "ofOpenFOAMCase_turboMachine" builder
    #
    
    self.dC["of"].runCurrentState()

  def Simulate(self):
    """Perform the simulation.

    Perform the simulation using foamlib. The simulation runs for 500 
    iterations as a laminar simulation. Afterwards, it is switched to turbulent
    mode.
    """
    cwd = os.getcwd()
    cpus_per_task = os.environ['SLURM_TRES_PER_TASK'].split('=')[-1]
    cpus_per_task = int(cpus_per_task)
    logger.info(f"Start CFD for state {self.state_} on {cpus_per_task} cores.")
    self.history_['Start Time'] = time.time()
    try:
        #
        # Create an FoamCase object of foamlib to control the simulation; the
        # "getDirectory" function returns the case directory that was created
        #
        fc = fl.FoamCase( 
          self.dC["of"].getDirectory(dtOO.lVHOstateHandler().commonState()) 
        )
        if cpus_per_task > 1:
            fc.decompose_par_dict['numberOfSubdomains'] = cpus_per_task 
            fc.decompose_par_dict['method'] = 'scotch' 
            fc.decompose_par()
        #
        # Turn off turbulence, modify the controlDict and run the simulation; this
        # is done twice
        #
        fc.turbulence_properties["RAS"]["turbulence"] = False
        fc.control_dict["endTime"] = 500
        fc.control_dict["writeInterval"] = 500
        if cpus_per_task > 1:
            fc.run(cmd=["mpiexec", "--oversubscribe", "-n", f"{cpus_per_task}","simpleFoam", "-parallel"])
        else:   
            fc.run() 
        fc.turbulence_properties["RAS"]["turbulence"] = True
        fc.control_dict["endTime"] = 1000
        fc.control_dict["writeInterval"] = 1000
        if cpus_per_task > 1:
            fc.run(cmd=["mpiexec", "--oversubscribe", "-n", f"{cpus_per_task}","simpleFoam", "-parallel"])
        else:
            fc.run()
        if cpus_per_task > 1:
            fc.reconstruct_par()
    except:
        logger.exception(f"Failed: {self.state_}.")
        
  
    


  @staticmethod
  def FailedFitness():
    """Failed fitness.

    Returns the value that represents a failed design.

    Returns
    -------
    float
      Fitness for a failed design.
    """
    return hydFoil.sys.float_info.max    

  def Evaluate(self):
    """Evaluate the simulation.

    The simulation is evaluated using the pyDtOO library. Additionally, the
    "patchToCsv" application is used to create csv files of the boundaries.

    The fitness function is calculated based on head deviation and efficiency.
    Efficiency is given by the equation

    .. math::
      
      \\Delta \\eta = 1 - \\frac{F_y u}{\\rho g H Q}

    with :math:`F_y`, :math:`u`, :math:`\\rho`, :math:`g`, :math:`H`, 
    and :math;`Q` that corresponds to force in :math:`y`-direction, rotational
    speed, density, gravitational constant, simulated head, and discharge. The 
    deviation in head is calculated by

    .. math::

      \\Delta H = \\frac{|H-H_d|}{H_d}

    with :math:`H_d` that corresponds to design head. Then, the fitness value 
    :math:`f` is defined as:

    .. math::

      f = \\Delta H + \\Delta \\eta \\mathrm{.}

    It is clear: the lower the fitness function :math:`f`, the better the
    candidate.

    Returns
    -------
    float:
      Fitness value of this candidate.
    """

    #
    # Get case directory of case
    #
    cDir = self.dC["of"].getDirectory(dtOO.lVHOstateHandler().commonState())
    self.cDir = cDir
    #logger.info(f"cDir: {cDir}.")
   
    #
    # Run "patchToCsv" to extract the data on the boundary; this executable
    # writes csv files for each boundary
    #
    subprocess.run(['patchToCsv', '-latestTime', 'U', 'INLET'], cwd=cDir)
    subprocess.run(['patchToCsv', '-latestTime', 'U', 'OUTLET'], cwd=cDir)
    subprocess.run(['patchToCsv', '-latestTime', 'p', 'INLET'], cwd=cDir)
    subprocess.run(['patchToCsv', '-latestTime', 'p', 'OUTLET'], cwd=cDir)
   
    #
    # Read the output generated by OpenFoam in the postProcessing folder; the
    # class automatically provides functions for calculating the average of
    # the forces
    #
    F = pd.dtForceDeveloping( 
      pd.dtDeveloping(cDir+'/postProcessing/F_BLADE').Read(
        {'force.dat' : ':,4:10', 'moment.dat' : ':,4:10', '*.*' : ''}
      )
    )

    #
    # Get the average of the force in y-direction
    #
    FMean = F.ForceMeanLast(10)[1]

    #
    # Initialize again an FoamCase object of foamlib
    #
    fc = fl.FoamCase(cDir)

    #
    # Get the last time directory
    #
    tName = fc[-1].name

    #
    # Read velocities and pressures at the inlet and outlet of the last time 
    # step; the class dtValueField provides functions to evaluate integral
    # values on the surfaces
    #
    U_i = pd.dtValueField( pd.dtField(cDir+'/INLET_U_'+tName+'.csv').Read() )
    U_o = pd.dtValueField( pd.dtField(cDir+'/OUTLET_U_'+tName+'.csv').Read() )
    p_i = pd.dtValueField( pd.dtField(cDir+'/INLET_p_'+tName+'.csv').Read() )
    p_o = pd.dtValueField( pd.dtField(cDir+'/OUTLET_p_'+tName+'.csv').Read() )
    
    #
    # Transform relative velocity "w" to absolute velocity "c" by subtracting
    # "u"
    #
    U_i.value_[:,1] = U_i.value_[:,1] + 2.0*np.pi*hydFoil.n_/60.*hydFoil.R_
    U_o.value_[:,1] = U_o.value_[:,1] + 2.0*np.pi*hydFoil.n_/60.*hydFoil.R_

    #
    # Set constants density and gravitational acceleration
    #
    rho = 997.0
    g = 9.81

    #
    # Integrate pressure and velocity over inlet and outlet; calculate the
    # sum of those energies to get the difference in head
    #
    e_i = (p_i.IntValueQ() / g + U_i.IntMagSquareQ() / 2.0 / g)
    e_o = (p_o.IntValueQ() / g + U_o.IntMagSquareQ() / 2.0 / g)
    Q_i = np.abs(U_i.IntQ())
    dHMean = (e_i + e_o) / Q_i
   
    #
    # Calculate efficiency of the blade
    #
    eta = np.abs((FMean * 2.0*np.pi*hydFoil.n_/60.*hydFoil.R_) / (rho*g*dHMean*Q_i))

    #
    # Output to log file
    #
    logging.info( 
      "dH = (e_i + e_o) / Q_i =( %f + %f ) / %f = %f / F = %f"
      % 
      (e_i, e_o, Q_i, dHMean, FMean) 
    )
    logging.info("eta = %f" % eta)

    #
    # Calculate the fitness value as an averaged sum of design head deviation
    # and efficiency
    #
    fit = np.abs(dHMean + 0.8)/0.8 + (1.0 - eta)

    #
    # Check if the simulated geometry is a pump or a turbine; if it is a pump,
    # return an artificial value that is greater than all other fitness values
    #
    failed_fit = min(10,max(10. * np.abs(fit),30))
    failed_turbine = False
    if (dHMean > 0.0) or (FMean < 0.0 ) or (eta > 1.0):
        fit = failed_fit
        failed_turbine = True
        turbine = 0
    if not failed_turbine:
        turbine = 1 # 1 -> correct turbine, -1 -> failed turbine ( pump etc.)
    else:
        logger.info(f"Pump detected: state {self.state_}.")
    #
    # Return fitness value
    #
    return fit, dHMean, FMean, eta


  def delete_case(self):
      label = "of_"
      dir_path = self.cDir
      try: 
          shutil.rmtree(dir_path)
          logger.debug(f"Deleted OpenFoam Case of state {self.state_} successfully.")
      except Exception as e:
          logger.warning(f"Failed deleting OpenFoam Case of state {self.state_}.")
          logger.warning(e)


  def get_history(self):
      return self.history_


def runHydFoil(x, state):
    try:
        hf = hydFoil( alpha_1=x[0], alpha_2=x[1], t_mid=x[2], state = state )
        state = hf.state_
        hf.Geometry()
        hf.GeometryMesh()
        start_time = time.time()
        hf.Mesh()
        end_time = time.time()
        logger.info(f"Proxy: finished meshing for state {state}. Time: {end_time - start_time}. Start CFD...")
        hf.Simulate()
        fit, dHMean, FMean, eta = hf.Evaluate()
        fit_extra = {'dHMean':dHMean.tolist(), 'FMean': float(FMean), 'eta': eta.tolist()}
        logger.info(f"STATE: {state}. SUCCESSFULLY.")
        hf.delete_case()
    except Exception as e:
        logger.warning("Proxy: Exception in runHydFoil: \n")
        logger.exception(e)
        fit = hydFoil.FailedFitness()
        fit_extra = {'dHMean': 1e6, 'FMean': 1e6, 'eta': 1e6}
    history = hf.get_history()
    if type(fit) == np.ndarray:
        fit = float(fit[0]) #serializable
    return fit, fit_extra, state, history



