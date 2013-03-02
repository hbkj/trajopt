#pragma once
#include <vector>
#include <map>
#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <openrave/openrave.h>

#include "sco/modeling.hpp"
#include "utils/basic_array.hpp"
#include "macros.h"

namespace trajopt {


namespace OR = OpenRAVE;
using OR::KinBody;
using OR::RobotBase;
using std::vector;
using std::map;
using namespace sco;
using namespace util;

typedef BasicArray<Var> VarArray;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DblMatrix;

typedef vector<double> DblVec;
typedef vector<int> IntVec;

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TrajArray;

/**
Interface for objects that know how to plot themselves given solution vector x
*/
class Plotter {
public:
  virtual void Plot(const DblVec& x, OR::EnvironmentBase&, std::vector<OR::GraphHandlePtr>& handles) = 0;
  virtual ~Plotter() {}
};

struct ObjectState {
  OpenRAVE::KinBodyPtr body;
  Vector3d xyz;
  Vector4d wxyz;
  DblVec dof_vals;
  IntVec dof_inds;
};
typedef boost::shared_ptr<ObjectState> ObjectStatePtr;
struct SceneState {
  int timestep;
  vector<ObjectStatePtr> obj_states;
};
typedef boost::shared_ptr<SceneState> SceneStatePtr;




}
