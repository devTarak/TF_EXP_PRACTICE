
¸
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
ś
AsString

input"T

output"
Ttype:
	2	
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ď
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.4.02v1.3.0-rc1-5916-g18c864cŁŃ

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step
VariableV2*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
o
input_example_tensorPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ParseExample/ConstConst*
_output_shapes
: *
valueB *
dtype0
b
ParseExample/ParseExample/namesConst*
_output_shapes
: *
valueB *
dtype0
h
&ParseExample/ParseExample/dense_keys_0Const*
_output_shapes
: *
value	B Bx*
dtype0
Ł
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names&ParseExample/ParseExample/dense_keys_0ParseExample/Const*
Tdense
2*
Ndense*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Nsparse *
sparse_types
 *
dense_shapes	
:

2dnn/input_from_feature_columns/input_layer/x/ShapeShapeParseExample/ParseExample*
T0*
out_type0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ú
:dnn/input_from_feature_columns/input_layer/x/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/x/Shape@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackBdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 

<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1Const*
value
B :*
dtype0*
_output_shapes
: 
ö
:dnn/input_from_feature_columns/input_layer/x/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/x/strided_slice<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
×
4dnn/input_from_feature_columns/input_layer/x/ReshapeReshapeParseExample/ParseExample:dnn/input_from_feature_columns/input_layer/x/Reshape/shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
~
<dnn/input_from_feature_columns/input_layer/concat/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ś
1dnn/input_from_feature_columns/input_layer/concatIdentity4dnn/input_from_feature_columns/input_layer/x/Reshape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"    *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *Ń_}˝*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *Ń_}=*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
 
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ž
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
 
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:
*
T0
Ë
dnn/hiddenlayer_0/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 

&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(
°
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0* 
_output_shapes
:

°
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes	
:
˝
dnn/hiddenlayer_0/bias/part_0
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
˙
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes	
:
Ľ
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:
u
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read* 
_output_shapes
:
*
T0
Č
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
l
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes	
:*
T0
 
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
h
dnn/zero_fraction/ConstConst*
_output_shapes
:*
valueB"       *
dtype0

dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values
Ť
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
Ĺ
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"    *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
ˇ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *W)˝*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *W)=*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
 
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0

>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ž
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
 
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0* 
_output_shapes
:

Ë
dnn/hiddenlayer_1/kernel/part_0
VariableV2* 
_output_shapes
:
*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape:
*
dtype0

&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
°
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0* 
_output_shapes
:
*
T0
°
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes	
:
˝
dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:
˙
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
Ľ
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes	
:
u
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read* 
_output_shapes
:
*
T0
­
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
l
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes	
:
 
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: *
T0
Ĺ
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"  Ä   *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
:
ˇ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *?Î˝*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *?Î=*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
 
Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
Ä*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
seed2 

>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
Ž
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0* 
_output_shapes
:
Ä
 
:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min* 
_output_shapes
:
Ä*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
Ë
dnn/hiddenlayer_2/kernel/part_0
VariableV2*
dtype0* 
_output_shapes
:
Ä*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
	container *
shape:
Ä

&dnn/hiddenlayer_2/kernel/part_0/AssignAssigndnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(* 
_output_shapes
:
Ä
°
$dnn/hiddenlayer_2/kernel/part_0/readIdentitydnn/hiddenlayer_2/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0* 
_output_shapes
:
Ä
°
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*
valueBÄ*    *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes	
:Ä
˝
dnn/hiddenlayer_2/bias/part_0
VariableV2*
shape:Ä*
dtype0*
_output_shapes	
:Ä*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
	container 
˙
$dnn/hiddenlayer_2/bias/part_0/AssignAssigndnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes	
:Ä
Ľ
"dnn/hiddenlayer_2/bias/part_0/readIdentitydnn/hiddenlayer_2/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes	
:Ä
u
dnn/hiddenlayer_2/kernelIdentity$dnn/hiddenlayer_2/kernel/part_0/read*
T0* 
_output_shapes
:
Ä
­
dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä*
transpose_a( *
transpose_b( 
l
dnn/hiddenlayer_2/biasIdentity"dnn/hiddenlayer_2/bias/part_0/read*
T0*
_output_shapes	
:Ä
 
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä*
T0
l
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
]
dnn/zero_fraction_2/zeroConst*
_output_shapes
: *
valueB
 *    *
dtype0

dnn/zero_fraction_2/EqualEqualdnn/hiddenlayer_2/Reludnn/zero_fraction_2/zero*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä*
T0
}
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä*

DstT0*

SrcT0

j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values
­
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
_output_shapes
: *
T0

$dnn/dnn/hiddenlayer_2/activation/tagConst*
dtype0*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_2/activation

 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
_output_shapes
: *
T0
ˇ
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"Ä      *+
_class!
loc:@dnn/logits/kernel/part_0
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *ľ2ž*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ľ2>*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0

Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	Ä*

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0
ţ
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@dnn/logits/kernel/part_0

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
:	Ä

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
:	Ä
ť
dnn/logits/kernel/part_0
VariableV2*
dtype0*
_output_shapes
:	Ä*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape:	Ä
ř
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes
:	Ä

dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
:	Ä
 
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
­
dnn/logits/bias/part_0
VariableV2*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
â
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0

dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
f
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes
:	Ä

dnn/logits/MatMulMatMuldnn/hiddenlayer_2/Reludnn/logits/kernel*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0

dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
dnn/zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
j
dnn/zero_fraction_3/ConstConst*
dtype0*
_output_shapes
:*
valueB"       

dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 

&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
_output_shapes
: *
T0
w
dnn/dnn/logits/activation/tagConst*
_output_shapes
: **
value!B Bdnn/dnn/logits/activation*
dtype0

dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: *
T0
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
5dnn/head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
g
_dnn/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
X
Pdnn/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
n
dnn/head/predictions/logisticSigmoiddnn/logits/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
dnn/head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
*dnn/head/predictions/two_class_logits/axisConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ů
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/logits/BiasAdd*dnn/head/predictions/two_class_logits/axis*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ć
dnn/head/predictions/class_idsArgMax%dnn/head/predictions/two_class_logits(dnn/head/predictions/class_ids/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
n
#dnn/head/predictions/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
°
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0	
Ý
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
T0	*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	precision˙˙˙˙˙˙˙˙˙*
shortest( 
p
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:
f
dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ś
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
V
dnn/head/range/startConst*
_output_shapes
: *
value	B : *
dtype0
V
dnn/head/range/limitConst*
value	B :*
dtype0*
_output_shapes
: 
V
dnn/head/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

dnn/head/rangeRangednn/head/range/startdnn/head/range/limitdnn/head/range/delta*

Tidx0*
_output_shapes
:
°
dnn/head/AsStringAsStringdnn/head/range*
_output_shapes
:*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0*

fill *

scientific( *
width˙˙˙˙˙˙˙˙˙
Y
dnn/head/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0

dnn/head/ExpandDims
ExpandDimsdnn/head/AsStringdnn/head/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
[
dnn/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 

dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:

dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_d2e286c9655a4e83a9b21efe86042de9/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
¸
save/SaveV2/tensor_namesConst"/device:CPU:0*Ü
valueŇBĎ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
ď
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB	B	784 0,784B784 784 0,784:0,784B	394 0,394B784 394 0,784:0,394B	196 0,196B394 196 0,394:0,196B1 0,1B196 1 0,196:0,1B *
dtype0*
_output_shapes
:	
˛
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2		
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
z
save/RestoreV2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:
q
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueBB	784 0,784*
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes	
:*
dtypes
2
Ĺ
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes	
:
~
save/RestoreV2_1/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
:
}
!save/RestoreV2_1/shape_and_slicesConst*(
valueBB784 784 0,784:0,784*
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices* 
_output_shapes
:
*
dtypes
2
Ň
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2_1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(* 
_output_shapes
:
*
use_locking(
|
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*+
value"B Bdnn/hiddenlayer_1/bias*
dtype0
s
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
valueBB	394 0,394*
dtype0

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes	
:*
dtypes
2
É
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2_2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes	
:
~
save/RestoreV2_3/tensor_namesConst*
dtype0*
_output_shapes
:*-
value$B"Bdnn/hiddenlayer_1/kernel
}
!save/RestoreV2_3/shape_and_slicesConst*(
valueBB784 394 0,784:0,394*
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices* 
_output_shapes
:
*
dtypes
2
Ň
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2_3* 
_output_shapes
:
*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(
|
save/RestoreV2_4/tensor_namesConst*+
value"B Bdnn/hiddenlayer_2/bias*
dtype0*
_output_shapes
:
s
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
valueBB	196 0,196*
dtype0

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes	
:Ä*
dtypes
2
É
save/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save/RestoreV2_4*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes	
:Ä*
use_locking(*
T0
~
save/RestoreV2_5/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/kernel*
dtype0*
_output_shapes
:
}
!save/RestoreV2_5/shape_and_slicesConst*(
valueBB394 196 0,394:0,196*
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices* 
_output_shapes
:
Ä*
dtypes
2
Ň
save/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save/RestoreV2_5*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(* 
_output_shapes
:
Ä*
use_locking(*
T0
u
save/RestoreV2_6/tensor_namesConst*$
valueBBdnn/logits/bias*
dtype0*
_output_shapes
:
o
!save/RestoreV2_6/shape_and_slicesConst*
valueBB1 0,1*
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
ş
save/Assign_6Assigndnn/logits/bias/part_0save/RestoreV2_6*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(
w
save/RestoreV2_7/tensor_namesConst*&
valueBBdnn/logits/kernel*
dtype0*
_output_shapes
:
y
!save/RestoreV2_7/shape_and_slicesConst*$
valueBB196 1 0,196:0,1*
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:	Ä
Ă
save/Assign_7Assigndnn/logits/kernel/part_0save/RestoreV2_7*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes
:	Ä
q
save/RestoreV2_8/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2	
 
save/Assign_8Assignglobal_stepsave/RestoreV2_8*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
¨
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp
+

group_depsNoOp^init^init_all_tables
R
save_1/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_473c24356c254a7caa47f61cc68f12fa/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
ş
save_1/SaveV2/tensor_namesConst"/device:CPU:0*Ü
valueŇBĎ	Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:	
ń
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB	B	784 0,784B784 784 0,784:0,784B	394 0,394B784 394 0,784:0,394B	196 0,196B394 196 0,394:0,196B1 0,1B196 1 0,196:0,1B *
dtype0*
_output_shapes
:	
ş
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2		
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
˛
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
_output_shapes
:*
T0*

axis *
N

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
|
save_1/RestoreV2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:
s
!save_1/RestoreV2/shape_and_slicesConst*
valueBB	784 0,784*
dtype0*
_output_shapes
:

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
_output_shapes	
:*
dtypes
2
É
save_1/AssignAssigndnn/hiddenlayer_0/bias/part_0save_1/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes	
:

save_1/RestoreV2_1/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
:

#save_1/RestoreV2_1/shape_and_slicesConst*(
valueBB784 784 0,784:0,784*
dtype0*
_output_shapes
:
Ś
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices* 
_output_shapes
:
*
dtypes
2
Ö
save_1/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save_1/RestoreV2_1*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(* 
_output_shapes
:

~
save_1/RestoreV2_2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_1/bias*
dtype0*
_output_shapes
:
u
#save_1/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
valueBB	394 0,394*
dtype0
Ą
save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes	
:
Í
save_1/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save_1/RestoreV2_2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes	
:

save_1/RestoreV2_3/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
:

#save_1/RestoreV2_3/shape_and_slicesConst*(
valueBB784 394 0,784:0,394*
dtype0*
_output_shapes
:
Ś
save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices* 
_output_shapes
:
*
dtypes
2
Ö
save_1/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save_1/RestoreV2_3*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(* 
_output_shapes
:

~
save_1/RestoreV2_4/tensor_namesConst*+
value"B Bdnn/hiddenlayer_2/bias*
dtype0*
_output_shapes
:
u
#save_1/RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueBB	196 0,196
Ą
save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
_output_shapes	
:Ä*
dtypes
2
Í
save_1/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save_1/RestoreV2_4*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes	
:Ä

save_1/RestoreV2_5/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/kernel*
dtype0*
_output_shapes
:

#save_1/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*(
valueBB394 196 0,394:0,196*
dtype0
Ś
save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices* 
_output_shapes
:
Ä*
dtypes
2
Ö
save_1/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save_1/RestoreV2_5*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(* 
_output_shapes
:
Ä
w
save_1/RestoreV2_6/tensor_namesConst*$
valueBBdnn/logits/bias*
dtype0*
_output_shapes
:
q
#save_1/RestoreV2_6/shape_and_slicesConst*
valueBB1 0,1*
dtype0*
_output_shapes
:
 
save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save_1/Assign_6Assigndnn/logits/bias/part_0save_1/RestoreV2_6*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(
y
save_1/RestoreV2_7/tensor_namesConst*&
valueBBdnn/logits/kernel*
dtype0*
_output_shapes
:
{
#save_1/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*$
valueBB196 1 0,196:0,1*
dtype0
Ľ
save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
_output_shapes
:	Ä*
dtypes
2
Ç
save_1/Assign_7Assigndnn/logits/kernel/part_0save_1/RestoreV2_7*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes
:	Ä*
use_locking(
s
save_1/RestoreV2_8/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2	
¤
save_1/Assign_8Assignglobal_stepsave_1/RestoreV2_8*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(
ź
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"×
	summariesÉ
Ć
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"ń
trainable_variablesŮÖ
Ý
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"*
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
Ý
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"*
dnn/hiddenlayer_1/kernel  "2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"#
dnn/hiddenlayer_1/bias "21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
Ý
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"*
dnn/hiddenlayer_2/kernelÄ  "Ä2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"#
dnn/hiddenlayer_2/biasÄ "Ä21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
¸
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"!
dnn/logits/kernelÄ  "Ä25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0" 
global_step

global_step:0"Á
	variablesł°
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
Ý
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"*
dnn/hiddenlayer_0/kernel  "2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"#
dnn/hiddenlayer_0/bias "21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
Ý
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"*
dnn/hiddenlayer_1/kernel  "2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"#
dnn/hiddenlayer_1/bias "21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
Ý
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"*
dnn/hiddenlayer_2/kernelÄ  "Ä2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
Ĺ
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"#
dnn/hiddenlayer_2/biasÄ "Ä21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
¸
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"!
dnn/logits/kernelÄ  "Ä25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0" 
legacy_init_op


group_deps*ß
classificationĚ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙1
classes&
dnn/head/Tile:0˙˙˙˙˙˙˙˙˙E
scores;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify*Ł

regression
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙A
outputs6
dnn/head/predictions/logistic:0˙˙˙˙˙˙˙˙˙tensorflow/serving/regress*ŕ
serving_defaultĚ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙E
scores;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙1
classes&
dnn/head/Tile:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify*ľ
predictŠ
5
examples)
input_example_tensor:0˙˙˙˙˙˙˙˙˙B
logistic6
dnn/head/predictions/logistic:0˙˙˙˙˙˙˙˙˙E
	class_ids8
!dnn/head/predictions/ExpandDims:0	˙˙˙˙˙˙˙˙˙L
probabilities;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙D
classes9
"dnn/head/predictions/str_classes:0˙˙˙˙˙˙˙˙˙5
logits+
dnn/logits/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict