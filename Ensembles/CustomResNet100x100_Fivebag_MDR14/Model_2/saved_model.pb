Є8
Њ§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8а0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
Ђ
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
: *
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
Ђ
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:@*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
Є
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:@*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:@*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:@*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:@*
dtype0
Є
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Є
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*о
valueгBЯ BЧ
Ъ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer-27
layer_with_weights-17
layer-28
	variables
regularization_losses
 trainable_variables
!	keras_api
"
signatures
 
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api

/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api

>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
R
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
R
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
h

Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
h

Ukernel
Vbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api

[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
aregularization_losses
btrainable_variables
c	keras_api
h

dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api

jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
R
s	variables
tregularization_losses
utrainable_variables
v	keras_api
R
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
i

{kernel
|bias
}	variables
~regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
V
Ѓ	variables
Єregularization_losses
Ѕtrainable_variables
І	keras_api
V
Ї	variables
Јregularization_losses
Љtrainable_variables
Њ	keras_api
V
Ћ	variables
Ќregularization_losses
­trainable_variables
Ў	keras_api
n
Џkernel
	Аbias
Б	variables
Вregularization_losses
Гtrainable_variables
Д	keras_api
V
Е	variables
Жregularization_losses
Зtrainable_variables
И	keras_api
n
Йkernel
	Кbias
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
V
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
n
Уkernel
	Фbias
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api

#0
$1
)2
*3
04
15
26
37
88
99
?10
@11
A12
B13
O14
P15
U16
V17
\18
]19
^20
_21
d22
e23
k24
l25
m26
n27
{28
|29
30
31
32
33
34
35
36
37
38
39
40
41
Џ42
А43
Й44
К45
У46
Ф47
 
Є
#0
$1
)2
*3
04
15
86
97
?8
@9
O10
P11
U12
V13
\14
]15
d16
e17
k18
l19
{20
|21
22
23
24
25
26
27
28
29
Џ30
А31
Й32
К33
У34
Ф35
В
	variables
regularization_losses
Щnon_trainable_variables
 trainable_variables
Ъlayer_metrics
 Ыlayer_regularization_losses
Ьlayers
Эmetrics
 
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
В
%	variables
&regularization_losses
Юnon_trainable_variables
'trainable_variables
Яlayer_metrics
 аlayer_regularization_losses
бlayers
вmetrics
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
В
+	variables
,regularization_losses
гnon_trainable_variables
-trainable_variables
дlayer_metrics
 еlayer_regularization_losses
жlayers
зmetrics
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33
 

00
11
В
4	variables
5regularization_losses
иnon_trainable_variables
6trainable_variables
йlayer_metrics
 кlayer_regularization_losses
лlayers
мmetrics
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
В
:	variables
;regularization_losses
нnon_trainable_variables
<trainable_variables
оlayer_metrics
 пlayer_regularization_losses
рlayers
сmetrics
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2
B3
 

?0
@1
В
C	variables
Dregularization_losses
тnon_trainable_variables
Etrainable_variables
уlayer_metrics
 фlayer_regularization_losses
хlayers
цmetrics
 
 
 
В
G	variables
Hregularization_losses
чnon_trainable_variables
Itrainable_variables
шlayer_metrics
 щlayer_regularization_losses
ъlayers
ыmetrics
 
 
 
В
K	variables
Lregularization_losses
ьnon_trainable_variables
Mtrainable_variables
эlayer_metrics
 юlayer_regularization_losses
яlayers
№metrics
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
В
Q	variables
Rregularization_losses
ёnon_trainable_variables
Strainable_variables
ђlayer_metrics
 ѓlayer_regularization_losses
єlayers
ѕmetrics
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
В
W	variables
Xregularization_losses
іnon_trainable_variables
Ytrainable_variables
їlayer_metrics
 јlayer_regularization_losses
љlayers
њmetrics
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
^2
_3
 

\0
]1
В
`	variables
aregularization_losses
ћnon_trainable_variables
btrainable_variables
ќlayer_metrics
 §layer_regularization_losses
ўlayers
џmetrics
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1
 

d0
e1
В
f	variables
gregularization_losses
non_trainable_variables
htrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
m2
n3
 

k0
l1
В
o	variables
pregularization_losses
non_trainable_variables
qtrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
 
 
 
В
s	variables
tregularization_losses
non_trainable_variables
utrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
 
 
 
В
w	variables
xregularization_losses
non_trainable_variables
ytrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
 

{0
|1
В
}	variables
~regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
][
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Е
	variables
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
 
hf
VARIABLE_VALUEbatch_normalization_10/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_10/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_10/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_10/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
Е
	variables
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
  layer_regularization_losses
Ёlayers
Ђmetrics
][
VARIABLE_VALUEconv2d_17/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_17/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Е
	variables
regularization_losses
Ѓnon_trainable_variables
trainable_variables
Єlayer_metrics
 Ѕlayer_regularization_losses
Іlayers
Їmetrics
 
hf
VARIABLE_VALUEbatch_normalization_11/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_11/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_11/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_11/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
Е
	variables
regularization_losses
Јnon_trainable_variables
trainable_variables
Љlayer_metrics
 Њlayer_regularization_losses
Ћlayers
Ќmetrics
 
 
 
Е
	variables
 regularization_losses
­non_trainable_variables
Ёtrainable_variables
Ўlayer_metrics
 Џlayer_regularization_losses
Аlayers
Бmetrics
 
 
 
Е
Ѓ	variables
Єregularization_losses
Вnon_trainable_variables
Ѕtrainable_variables
Гlayer_metrics
 Дlayer_regularization_losses
Еlayers
Жmetrics
 
 
 
Е
Ї	variables
Јregularization_losses
Зnon_trainable_variables
Љtrainable_variables
Иlayer_metrics
 Йlayer_regularization_losses
Кlayers
Лmetrics
 
 
 
Е
Ћ	variables
Ќregularization_losses
Мnon_trainable_variables
­trainable_variables
Нlayer_metrics
 Оlayer_regularization_losses
Пlayers
Рmetrics
[Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

Џ0
А1
 

Џ0
А1
Е
Б	variables
Вregularization_losses
Сnon_trainable_variables
Гtrainable_variables
Тlayer_metrics
 Уlayer_regularization_losses
Фlayers
Хmetrics
 
 
 
Е
Е	variables
Жregularization_losses
Цnon_trainable_variables
Зtrainable_variables
Чlayer_metrics
 Шlayer_regularization_losses
Щlayers
Ъmetrics
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

Й0
К1
 

Й0
К1
Е
Л	variables
Мregularization_losses
Ыnon_trainable_variables
Нtrainable_variables
Ьlayer_metrics
 Эlayer_regularization_losses
Юlayers
Яmetrics
 
 
 
Е
П	variables
Рregularization_losses
аnon_trainable_variables
Сtrainable_variables
бlayer_metrics
 вlayer_regularization_losses
гlayers
дmetrics
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

У0
Ф1
 

У0
Ф1
Е
Х	variables
Цregularization_losses
еnon_trainable_variables
Чtrainable_variables
жlayer_metrics
 зlayer_regularization_losses
иlayers
йmetrics
Z
20
31
A2
B3
^4
_5
m6
n7
8
9
10
11
 
 
о
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 
 
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

^0
_1
 
 
 
 
 
 
 
 
 

m0
n1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_2Placeholder*/
_output_shapes
:џџџџџџџџџ22*
dtype0*$
shape:џџџџџџџџџ22

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*<
Tin5
321*
Tout
2*'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_237816
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*=
Tin6
422*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_240740
Ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*<
Tin5
321*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_240896џ-
ї
Ћ
C__inference_dense_5_layer_call_and_return_conditional_losses_240248

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
SigmoidФ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addМ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ъ
m
__inference_loss_fn_23_240569;
7dense_5_bias_regularizer_square_readvariableop_resource
identityд
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_5_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addc
IdentityIdentity dense_5/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ј
Њ
7__inference_batch_normalization_11_layer_call_fn_239926

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2351972
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Џ
o
__inference_loss_fn_22_240556=
9dense_5_kernel_regularizer_square_readvariableop_resource
identityп
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_5_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/adde
IdentityIdentity"dense_5/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ш

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239200

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_240195

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_240111

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
Ћ
C__inference_dense_3_layer_call_and_return_conditional_losses_235931

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluФ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addН
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239022

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
­
E__inference_conv2d_17_layer_call_and_return_conditional_losses_235103

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddЯ
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addР
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
і
Љ
6__inference_batch_normalization_6_layer_call_fn_239035

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2343042
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
љ
}
(__inference_dense_3_layer_call_fn_240099

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2359312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ш

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238947

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
$
и
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239182

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_235525

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
$
и
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_235507

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
љ
q
__inference_loss_fn_12_240426?
;conv2d_15_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_15_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addg
IdentityIdentity$conv2d_15/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ў
Љ
6__inference_batch_normalization_7_layer_call_fn_239213

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2353852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ђ
R
&__inference_add_3_layer_call_fn_239238
inputs_0
inputs_1
identityЕ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_2354452
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22:џџџџџџџџџ22:Y U
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/1
Щ

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239988

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у

*__inference_conv2d_14_layer_call_fn_234748

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2347382
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ў
Љ
6__inference_batch_normalization_9_layer_call_fn_239607

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2355962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ј
Њ
7__inference_batch_normalization_10_layer_call_fn_239748

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2350342
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о
o
__inference_loss_fn_0_240270>
:conv2d_9_kernel_regularizer_square_readvariableop_resource
identityщ
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_9_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addf
IdentityIdentity#conv2d_9/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 


R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_235228

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ћ
Џ
C__inference_model_1_layer_call_and_return_conditional_losses_236929

inputs
conv2d_9_236612
conv2d_9_236614
conv2d_10_236617
conv2d_10_236619 
batch_normalization_6_236622 
batch_normalization_6_236624 
batch_normalization_6_236626 
batch_normalization_6_236628
conv2d_11_236631
conv2d_11_236633 
batch_normalization_7_236636 
batch_normalization_7_236638 
batch_normalization_7_236640 
batch_normalization_7_236642
conv2d_12_236647
conv2d_12_236649
conv2d_13_236652
conv2d_13_236654 
batch_normalization_8_236657 
batch_normalization_8_236659 
batch_normalization_8_236661 
batch_normalization_8_236663
conv2d_14_236666
conv2d_14_236668 
batch_normalization_9_236671 
batch_normalization_9_236673 
batch_normalization_9_236675 
batch_normalization_9_236677
conv2d_15_236682
conv2d_15_236684
conv2d_16_236687
conv2d_16_236689!
batch_normalization_10_236692!
batch_normalization_10_236694!
batch_normalization_10_236696!
batch_normalization_10_236698
conv2d_17_236701
conv2d_17_236703!
batch_normalization_11_236706!
batch_normalization_11_236708!
batch_normalization_11_236710!
batch_normalization_11_236712
dense_3_236719
dense_3_236721
dense_4_236725
dense_4_236727
dense_5_236731
dense_5_236733
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCall§
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_236612conv2d_9_236614*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2341722"
 conv2d_9/StatefulPartitionedCallЅ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_236617conv2d_10_236619*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2342102#
!conv2d_10/StatefulPartitionedCall 
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_236622batch_normalization_6_236624batch_normalization_6_236626batch_normalization_6_236628*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2352962/
-batch_normalization_6/StatefulPartitionedCallВ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_236631conv2d_11_236633*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2343732#
!conv2d_11/StatefulPartitionedCall 
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_236636batch_normalization_7_236638batch_normalization_7_236640batch_normalization_7_236642*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2353852/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_2354452
add_3/PartitionedCallс
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_2354592
activation_3/PartitionedCallЁ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_236647conv2d_12_236649*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2345372#
!conv2d_12/StatefulPartitionedCallІ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_236652conv2d_13_236654*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2345752#
!conv2d_13/StatefulPartitionedCall 
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_236657batch_normalization_8_236659batch_normalization_8_236661batch_normalization_8_236663*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2355072/
-batch_normalization_8/StatefulPartitionedCallВ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_236666conv2d_14_236668*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2347382#
!conv2d_14/StatefulPartitionedCall 
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_236671batch_normalization_9_236673batch_normalization_9_236675batch_normalization_9_236677*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2355962/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2356562
add_4/PartitionedCallс
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_2356702
activation_4/PartitionedCallЁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_236682conv2d_15_236684*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2349022#
!conv2d_15/StatefulPartitionedCallІ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_236687conv2d_16_236689*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2349402#
!conv2d_16/StatefulPartitionedCallЇ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_236692batch_normalization_10_236694batch_normalization_10_236696batch_normalization_10_236698*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23571820
.batch_normalization_10/StatefulPartitionedCallГ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_236701conv2d_17_236703*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2351032#
!conv2d_17/StatefulPartitionedCallЇ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_236706batch_normalization_11_236708batch_normalization_11_236710batch_normalization_11_236712*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23580720
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2358672
add_5/PartitionedCallс
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_2358812
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_2352462,
*global_average_pooling2d_1/PartitionedCallх
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2358962
flatten_1/PartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_236719dense_3_236721*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2359312!
dense_3/StatefulPartitionedCallѓ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2359592#
!dropout_2/StatefulPartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_236725dense_4_236727*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2360042!
dense_4/StatefulPartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360322#
!dropout_3/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_236731dense_5_236733*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2360772!
dense_5/StatefulPartitionedCallО
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_236612*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЎ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_236614*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addС
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_236617*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addБ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_236619*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addС
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_236631*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addБ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_236633*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addС
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_236647*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addБ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_236649*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addС
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_236652*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addБ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_236654*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addС
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_236666*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addБ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_236668*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addС
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_236682*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addБ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_236684*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addС
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_236687*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addБ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_236689*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addС
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_236701*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addБ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_236703*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addД
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_236719*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЌ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_236721*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЕ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_236725* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЌ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_236727*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addД
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_236731*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЋ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_236733*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/add
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
пе
У
!__inference__wrapped_model_234145
input_23
/model_1_conv2d_9_conv2d_readvariableop_resource4
0model_1_conv2d_9_biasadd_readvariableop_resource4
0model_1_conv2d_10_conv2d_readvariableop_resource5
1model_1_conv2d_10_biasadd_readvariableop_resource9
5model_1_batch_normalization_6_readvariableop_resource;
7model_1_batch_normalization_6_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_11_conv2d_readvariableop_resource5
1model_1_conv2d_11_biasadd_readvariableop_resource9
5model_1_batch_normalization_7_readvariableop_resource;
7model_1_batch_normalization_7_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_12_conv2d_readvariableop_resource5
1model_1_conv2d_12_biasadd_readvariableop_resource4
0model_1_conv2d_13_conv2d_readvariableop_resource5
1model_1_conv2d_13_biasadd_readvariableop_resource9
5model_1_batch_normalization_8_readvariableop_resource;
7model_1_batch_normalization_8_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_14_conv2d_readvariableop_resource5
1model_1_conv2d_14_biasadd_readvariableop_resource9
5model_1_batch_normalization_9_readvariableop_resource;
7model_1_batch_normalization_9_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_15_conv2d_readvariableop_resource5
1model_1_conv2d_15_biasadd_readvariableop_resource4
0model_1_conv2d_16_conv2d_readvariableop_resource5
1model_1_conv2d_16_biasadd_readvariableop_resource:
6model_1_batch_normalization_10_readvariableop_resource<
8model_1_batch_normalization_10_readvariableop_1_resourceK
Gmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_17_conv2d_readvariableop_resource5
1model_1_conv2d_17_biasadd_readvariableop_resource:
6model_1_batch_normalization_11_readvariableop_resource<
8model_1_batch_normalization_11_readvariableop_1_resourceK
Gmodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceM
Imodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource2
.model_1_dense_3_matmul_readvariableop_resource3
/model_1_dense_3_biasadd_readvariableop_resource2
.model_1_dense_4_matmul_readvariableop_resource3
/model_1_dense_4_biasadd_readvariableop_resource2
.model_1_dense_5_matmul_readvariableop_resource3
/model_1_dense_5_biasadd_readvariableop_resource
identityШ
&model_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/conv2d_9/Conv2D/ReadVariableOpз
model_1/conv2d_9/Conv2DConv2Dinput_2.model_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_1/conv2d_9/Conv2DП
'model_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_9/BiasAdd/ReadVariableOpЬ
model_1/conv2d_9/BiasAddBiasAdd model_1/conv2d_9/Conv2D:output:0/model_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_9/BiasAddЫ
'model_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_10/Conv2D/ReadVariableOpє
model_1/conv2d_10/Conv2DConv2D!model_1/conv2d_9/BiasAdd:output:0/model_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_1/conv2d_10/Conv2DТ
(model_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_10/BiasAdd/ReadVariableOpа
model_1/conv2d_10/BiasAddBiasAdd!model_1/conv2d_10/Conv2D:output:00model_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_10/BiasAdd
model_1/conv2d_10/ReluRelu"model_1/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_10/ReluЮ
,model_1/batch_normalization_6/ReadVariableOpReadVariableOp5model_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_1/batch_normalization_6/ReadVariableOpд
.model_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype020
.model_1/batch_normalization_6/ReadVariableOp_1
=model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?model_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$model_1/conv2d_10/Relu:activations:04model_1/batch_normalization_6/ReadVariableOp:value:06model_1/batch_normalization_6/ReadVariableOp_1:value:0Emodel_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_6/FusedBatchNormV3Ы
'model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_11/Conv2D/ReadVariableOp
model_1/conv2d_11/Conv2DConv2D2model_1/batch_normalization_6/FusedBatchNormV3:y:0/model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
model_1/conv2d_11/Conv2DТ
(model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_11/BiasAdd/ReadVariableOpа
model_1/conv2d_11/BiasAddBiasAdd!model_1/conv2d_11/Conv2D:output:00model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/conv2d_11/BiasAddЮ
,model_1/batch_normalization_7/ReadVariableOpReadVariableOp5model_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_1/batch_normalization_7/ReadVariableOpд
.model_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype020
.model_1/batch_normalization_7/ReadVariableOp_1
=model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?model_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_11/BiasAdd:output:04model_1/batch_normalization_7/ReadVariableOp:value:06model_1/batch_normalization_7/ReadVariableOp_1:value:0Emodel_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_7/FusedBatchNormV3Р
model_1/add_3/addAddV22model_1/batch_normalization_7/FusedBatchNormV3:y:0!model_1/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/add_3/add
model_1/activation_3/ReluRelumodel_1/add_3/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
model_1/activation_3/ReluЫ
'model_1/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_1/conv2d_12/Conv2D/ReadVariableOpњ
model_1/conv2d_12/Conv2DConv2D'model_1/activation_3/Relu:activations:0/model_1/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_1/conv2d_12/Conv2DТ
(model_1/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_12/BiasAdd/ReadVariableOpа
model_1/conv2d_12/BiasAddBiasAdd!model_1/conv2d_12/Conv2D:output:00model_1/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_12/BiasAdd
model_1/conv2d_12/ReluRelu"model_1/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_12/ReluЫ
'model_1/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_1/conv2d_13/Conv2D/ReadVariableOpї
model_1/conv2d_13/Conv2DConv2D$model_1/conv2d_12/Relu:activations:0/model_1/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_1/conv2d_13/Conv2DТ
(model_1/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_13/BiasAdd/ReadVariableOpа
model_1/conv2d_13/BiasAddBiasAdd!model_1/conv2d_13/Conv2D:output:00model_1/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_13/BiasAdd
model_1/conv2d_13/ReluRelu"model_1/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_13/ReluЮ
,model_1/batch_normalization_8/ReadVariableOpReadVariableOp5model_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_8/ReadVariableOpд
.model_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_8/ReadVariableOp_1
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$model_1/conv2d_13/Relu:activations:04model_1/batch_normalization_8/ReadVariableOp:value:06model_1/batch_normalization_8/ReadVariableOp_1:value:0Emodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_8/FusedBatchNormV3Ы
'model_1/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_1/conv2d_14/Conv2D/ReadVariableOp
model_1/conv2d_14/Conv2DConv2D2model_1/batch_normalization_8/FusedBatchNormV3:y:0/model_1/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
model_1/conv2d_14/Conv2DТ
(model_1/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_14/BiasAdd/ReadVariableOpа
model_1/conv2d_14/BiasAddBiasAdd!model_1/conv2d_14/Conv2D:output:00model_1/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/conv2d_14/BiasAddЮ
,model_1/batch_normalization_9/ReadVariableOpReadVariableOp5model_1_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_1/batch_normalization_9/ReadVariableOpд
.model_1/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_1/batch_normalization_9/ReadVariableOp_1
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02?
=model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02A
?model_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1
.model_1/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_14/BiasAdd:output:04model_1/batch_normalization_9/ReadVariableOp:value:06model_1/batch_normalization_9/ReadVariableOp_1:value:0Emodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 20
.model_1/batch_normalization_9/FusedBatchNormV3У
model_1/add_4/addAddV22model_1/batch_normalization_9/FusedBatchNormV3:y:0$model_1/conv2d_12/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/add_4/add
model_1/activation_4/ReluRelumodel_1/add_4/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
model_1/activation_4/ReluЫ
'model_1/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_1/conv2d_15/Conv2D/ReadVariableOpњ
model_1/conv2d_15/Conv2DConv2D'model_1/activation_4/Relu:activations:0/model_1/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_1/conv2d_15/Conv2DТ
(model_1/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_15/BiasAdd/ReadVariableOpа
model_1/conv2d_15/BiasAddBiasAdd!model_1/conv2d_15/Conv2D:output:00model_1/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_15/BiasAdd
model_1/conv2d_15/ReluRelu"model_1/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_15/ReluЫ
'model_1/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_1/conv2d_16/Conv2D/ReadVariableOpї
model_1/conv2d_16/Conv2DConv2D$model_1/conv2d_15/Relu:activations:0/model_1/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_1/conv2d_16/Conv2DТ
(model_1/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_16/BiasAdd/ReadVariableOpа
model_1/conv2d_16/BiasAddBiasAdd!model_1/conv2d_16/Conv2D:output:00model_1/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_16/BiasAdd
model_1/conv2d_16/ReluRelu"model_1/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_16/Reluб
-model_1/batch_normalization_10/ReadVariableOpReadVariableOp6model_1_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_10/ReadVariableOpз
/model_1/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_10/ReadVariableOp_1
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ђ
/model_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3$model_1/conv2d_16/Relu:activations:05model_1/batch_normalization_10/ReadVariableOp:value:07model_1/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 21
/model_1/batch_normalization_10/FusedBatchNormV3Ы
'model_1/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_1/conv2d_17/Conv2D/ReadVariableOp
model_1/conv2d_17/Conv2DConv2D3model_1/batch_normalization_10/FusedBatchNormV3:y:0/model_1/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
model_1/conv2d_17/Conv2DТ
(model_1/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_17/BiasAdd/ReadVariableOpа
model_1/conv2d_17/BiasAddBiasAdd!model_1/conv2d_17/Conv2D:output:00model_1/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/conv2d_17/BiasAddб
-model_1/batch_normalization_11/ReadVariableOpReadVariableOp6model_1_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/batch_normalization_11/ReadVariableOpз
/model_1/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_1_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_1/batch_normalization_11/ReadVariableOp_1
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1 
/model_1/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3"model_1/conv2d_17/BiasAdd:output:05model_1/batch_normalization_11/ReadVariableOp:value:07model_1/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 21
/model_1/batch_normalization_11/FusedBatchNormV3Ф
model_1/add_5/addAddV23model_1/batch_normalization_11/FusedBatchNormV3:y:0$model_1/conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/add_5/add
model_1/activation_5/ReluRelumodel_1/add_5/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
model_1/activation_5/ReluЧ
9model_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/global_average_pooling2d_1/Mean/reduction_indicesљ
'model_1/global_average_pooling2d_1/MeanMean'model_1/activation_5/Relu:activations:0Bmodel_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2)
'model_1/global_average_pooling2d_1/Mean
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
model_1/flatten_1/ConstЧ
model_1/flatten_1/ReshapeReshape0model_1/global_average_pooling2d_1/Mean:output:0 model_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_1/flatten_1/ReshapeО
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02'
%model_1/dense_3/MatMul/ReadVariableOpР
model_1/dense_3/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_3/MatMulН
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOpТ
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_3/BiasAdd
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_3/Relu
model_1/dropout_2/IdentityIdentity"model_1/dense_3/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dropout_2/IdentityП
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%model_1/dense_4/MatMul/ReadVariableOpС
model_1/dense_4/MatMulMatMul#model_1/dropout_2/Identity:output:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_4/MatMulН
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOpТ
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_4/BiasAdd
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dense_4/Relu
model_1/dropout_3/IdentityIdentity"model_1/dense_4/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_1/dropout_3/IdentityО
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%model_1/dense_5/MatMul/ReadVariableOpР
model_1/dense_5/MatMulMatMul#model_1/dropout_3/Identity:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_1/dense_5/MatMulМ
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOpС
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_1/dense_5/BiasAdd
model_1/dense_5/SigmoidSigmoid model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_1/dense_5/Sigmoido
IdentityIdentitymodel_1/dense_5/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22:::::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
у

*__inference_conv2d_11_layer_call_fn_234383

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2343732
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ј
Љ
6__inference_batch_normalization_8_layer_call_fn_239367

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2347002
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 $
й
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_235807

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ј
p
__inference_loss_fn_4_240322?
;conv2d_11_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_11_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addg
IdentityIdentity$conv2d_11/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ј
Љ
6__inference_batch_normalization_7_layer_call_fn_239151

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2344982
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

n
__inference_loss_fn_9_240387=
9conv2d_13_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_13_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/adde
IdentityIdentity"conv2d_13/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_235959

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_234335

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж
d
H__inference_activation_3_layer_call_and_return_conditional_losses_239243

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
Ш

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239416

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
­
E__inference_conv2d_14_layer_call_and_return_conditional_losses_234738

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddЯ
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addР
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
М
­
E__inference_conv2d_11_layer_call_and_return_conditional_losses_234373

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddЯ
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addР
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
у

*__inference_conv2d_15_layer_call_fn_234912

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2349022
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч$
и
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_234467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г 
­
E__inference_conv2d_12_layer_call_and_return_conditional_losses_234537

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluЯ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addР
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239341

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч$
и
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_235065

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

c
*__inference_dropout_2_layer_call_fn_240121

inputs
identityЂStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2359592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_240190

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј
p
__inference_loss_fn_6_240348?
;conv2d_12_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_12_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addg
IdentityIdentity$conv2d_12/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
з
m
A__inference_add_4_layer_call_and_return_conditional_losses_239626
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22 :џџџџџџџџџ22 :Y U
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/1
љ
F
*__inference_dropout_2_layer_call_fn_240126

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2359642
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч$
и
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ћ
}
(__inference_dense_4_layer_call_fn_240178

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2360042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
$
и
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239398

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А
Љ
6__inference_batch_normalization_6_layer_call_fn_238973

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2353142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я
k
A__inference_add_4_layer_call_and_return_conditional_losses_235656

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22 :џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs
ј
Љ
6__inference_batch_normalization_9_layer_call_fn_239545

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2348632
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
љ
F
*__inference_dropout_3_layer_call_fn_240205

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
Љ
6__inference_batch_normalization_7_layer_call_fn_239138

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2344672
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ћ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_235896

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ЃЮ

"__inference__traced_restore_240896
file_prefix$
 assignvariableop_conv2d_9_kernel$
 assignvariableop_1_conv2d_9_bias'
#assignvariableop_2_conv2d_10_kernel%
!assignvariableop_3_conv2d_10_bias2
.assignvariableop_4_batch_normalization_6_gamma1
-assignvariableop_5_batch_normalization_6_beta8
4assignvariableop_6_batch_normalization_6_moving_mean<
8assignvariableop_7_batch_normalization_6_moving_variance'
#assignvariableop_8_conv2d_11_kernel%
!assignvariableop_9_conv2d_11_bias3
/assignvariableop_10_batch_normalization_7_gamma2
.assignvariableop_11_batch_normalization_7_beta9
5assignvariableop_12_batch_normalization_7_moving_mean=
9assignvariableop_13_batch_normalization_7_moving_variance(
$assignvariableop_14_conv2d_12_kernel&
"assignvariableop_15_conv2d_12_bias(
$assignvariableop_16_conv2d_13_kernel&
"assignvariableop_17_conv2d_13_bias3
/assignvariableop_18_batch_normalization_8_gamma2
.assignvariableop_19_batch_normalization_8_beta9
5assignvariableop_20_batch_normalization_8_moving_mean=
9assignvariableop_21_batch_normalization_8_moving_variance(
$assignvariableop_22_conv2d_14_kernel&
"assignvariableop_23_conv2d_14_bias3
/assignvariableop_24_batch_normalization_9_gamma2
.assignvariableop_25_batch_normalization_9_beta9
5assignvariableop_26_batch_normalization_9_moving_mean=
9assignvariableop_27_batch_normalization_9_moving_variance(
$assignvariableop_28_conv2d_15_kernel&
"assignvariableop_29_conv2d_15_bias(
$assignvariableop_30_conv2d_16_kernel&
"assignvariableop_31_conv2d_16_bias4
0assignvariableop_32_batch_normalization_10_gamma3
/assignvariableop_33_batch_normalization_10_beta:
6assignvariableop_34_batch_normalization_10_moving_mean>
:assignvariableop_35_batch_normalization_10_moving_variance(
$assignvariableop_36_conv2d_17_kernel&
"assignvariableop_37_conv2d_17_bias4
0assignvariableop_38_batch_normalization_11_gamma3
/assignvariableop_39_batch_normalization_11_beta:
6assignvariableop_40_batch_normalization_11_moving_mean>
:assignvariableop_41_batch_normalization_11_moving_variance&
"assignvariableop_42_dense_3_kernel$
 assignvariableop_43_dense_3_bias&
"assignvariableop_44_dense_4_kernel$
 assignvariableop_45_dense_4_bias&
"assignvariableop_46_dense_5_kernel$
 assignvariableop_47_dense_5_bias
identity_49ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1Ч
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*г
valueЩBЦ0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesю
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ж
_output_shapesУ
Р::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
2202
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_9_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_9_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_10_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_10_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Є
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_6_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_6_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Њ
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_6_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ў
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_6_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_11_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_11_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Ј
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_7_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ї
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_7_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ў
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_7_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13В
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_7_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_12_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_12_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_13_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_13_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ј
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_8_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ї
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_8_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ў
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_8_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_8_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_14_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_14_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ј
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_9_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ї
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_9_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ў
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_9_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27В
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_9_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_15_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_15_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_16_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_16_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Љ
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_10_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ј
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_10_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Џ
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_10_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Г
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_10_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_17_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_17_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Љ
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_11_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39Ј
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_11_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Џ
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_11_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_11_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_3_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_3_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_4_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_4_biasIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_5_kernelIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_5_biasIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpў
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48	
Identity_49IdentityIdentity_48:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_49"#
identity_49Identity_49:output:0*з
_input_shapesХ
Т: ::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ј
p
__inference_loss_fn_2_240296?
;conv2d_10_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_10_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addg
IdentityIdentity$conv2d_10/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ћ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_240042

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

o
__inference_loss_fn_17_240491=
9conv2d_17_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_17_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/adde
IdentityIdentity"conv2d_17/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Щ

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_235736

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

n
__inference_loss_fn_5_240335=
9conv2d_11_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_11_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/adde
IdentityIdentity"conv2d_11/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
В
Њ
7__inference_batch_normalization_11_layer_call_fn_240014

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2358252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж
б
$__inference_signature_wrapper_237816
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_2341452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 


Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_234700

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
і
Љ
6__inference_batch_normalization_9_layer_call_fn_239532

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2348322
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239735

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж
d
H__inference_activation_5_layer_call_and_return_conditional_losses_235881

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs
А
Љ
6__inference_batch_normalization_9_layer_call_fn_239620

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2356142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
і
Љ
6__inference_batch_normalization_8_layer_call_fn_239354

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2346692
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
љ
q
__inference_loss_fn_10_240400?
;conv2d_14_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_14_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addg
IdentityIdentity$conv2d_14/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ї
Ћ
C__inference_dense_5_layer_call_and_return_conditional_losses_236077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
SigmoidФ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addМ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_234498

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

o
__inference_loss_fn_15_240465=
9conv2d_16_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_16_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/adde
IdentityIdentity"conv2d_16/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 

m
__inference_loss_fn_1_240283<
8conv2d_9_bias_regularizer_square_readvariableop_resource
identityз
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_9_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addd
IdentityIdentity!conv2d_9/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ђ
R
&__inference_add_5_layer_call_fn_240026
inputs_0
inputs_1
identityЕ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2358672
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22@:џџџџџџџџџ22@:Y U
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/1
Ш

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_235314

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
$
и
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_235385

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239913

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
Ћ
C__inference_dense_3_layer_call_and_return_conditional_losses_240090

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluФ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addН
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ў
Љ
6__inference_batch_normalization_6_layer_call_fn_238960

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2352962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_236037

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
m
__inference_loss_fn_19_240517;
7dense_3_bias_regularizer_square_readvariableop_resource
identityе
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_3_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addc
IdentityIdentity dense_3/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Я
k
A__inference_add_3_layer_call_and_return_conditional_losses_235445

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
Щ

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_235825

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

I
-__inference_activation_5_layer_call_fn_240036

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_2358812
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs
њ
Њ
7__inference_batch_normalization_10_layer_call_fn_239761

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2350652
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з
m
A__inference_add_3_layer_call_and_return_conditional_losses_239232
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22:џџџџџџџџџ22:Y U
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22
"
_user_specified_name
inputs/1
ч$
и
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239323

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

n
__inference_loss_fn_7_240361=
9conv2d_12_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_12_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/adde
IdentityIdentity"conv2d_12/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_236032

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш$
й
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_235197

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у

*__inference_conv2d_16_layer_call_fn_234950

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2349402
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
э
д
(__inference_model_1_layer_call_fn_238753

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2369292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ј
Љ
6__inference_batch_normalization_6_layer_call_fn_239048

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2343352
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч$
и
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239004

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѕ
F
*__inference_flatten_1_layer_call_fn_240047

inputs
identityЄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2358962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

I
-__inference_activation_3_layer_call_fn_239248

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_2354592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
В
Њ
7__inference_batch_normalization_10_layer_call_fn_239836

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2357362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
o
__inference_loss_fn_20_240530=
9dense_4_kernel_regularizer_square_readvariableop_resource
identityр
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_4_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/adde
IdentityIdentity"dense_4/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Г 
­
E__inference_conv2d_13_layer_call_and_return_conditional_losses_234575

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluЯ
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addР
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
М
r
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_235246

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ч$
и
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_234304

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж
d
H__inference_activation_4_layer_call_and_return_conditional_losses_235670

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs

ш
C__inference_model_1_layer_call_and_return_conditional_losses_236606
input_2
conv2d_9_236289
conv2d_9_236291
conv2d_10_236294
conv2d_10_236296 
batch_normalization_6_236299 
batch_normalization_6_236301 
batch_normalization_6_236303 
batch_normalization_6_236305
conv2d_11_236308
conv2d_11_236310 
batch_normalization_7_236313 
batch_normalization_7_236315 
batch_normalization_7_236317 
batch_normalization_7_236319
conv2d_12_236324
conv2d_12_236326
conv2d_13_236329
conv2d_13_236331 
batch_normalization_8_236334 
batch_normalization_8_236336 
batch_normalization_8_236338 
batch_normalization_8_236340
conv2d_14_236343
conv2d_14_236345 
batch_normalization_9_236348 
batch_normalization_9_236350 
batch_normalization_9_236352 
batch_normalization_9_236354
conv2d_15_236359
conv2d_15_236361
conv2d_16_236364
conv2d_16_236366!
batch_normalization_10_236369!
batch_normalization_10_236371!
batch_normalization_10_236373!
batch_normalization_10_236375
conv2d_17_236378
conv2d_17_236380!
batch_normalization_11_236383!
batch_normalization_11_236385!
batch_normalization_11_236387!
batch_normalization_11_236389
dense_3_236396
dense_3_236398
dense_4_236402
dense_4_236404
dense_5_236408
dense_5_236410
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallў
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_9_236289conv2d_9_236291*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2341722"
 conv2d_9/StatefulPartitionedCallЅ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_236294conv2d_10_236296*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2342102#
!conv2d_10/StatefulPartitionedCallЂ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_236299batch_normalization_6_236301batch_normalization_6_236303batch_normalization_6_236305*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2353142/
-batch_normalization_6/StatefulPartitionedCallВ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_236308conv2d_11_236310*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2343732#
!conv2d_11/StatefulPartitionedCallЂ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_236313batch_normalization_7_236315batch_normalization_7_236317batch_normalization_7_236319*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2354032/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_2354452
add_3/PartitionedCallс
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_2354592
activation_3/PartitionedCallЁ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_236324conv2d_12_236326*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2345372#
!conv2d_12/StatefulPartitionedCallІ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_236329conv2d_13_236331*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2345752#
!conv2d_13/StatefulPartitionedCallЂ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_236334batch_normalization_8_236336batch_normalization_8_236338batch_normalization_8_236340*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2355252/
-batch_normalization_8/StatefulPartitionedCallВ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_236343conv2d_14_236345*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2347382#
!conv2d_14/StatefulPartitionedCallЂ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_236348batch_normalization_9_236350batch_normalization_9_236352batch_normalization_9_236354*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2356142/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2356562
add_4/PartitionedCallс
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_2356702
activation_4/PartitionedCallЁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_236359conv2d_15_236361*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2349022#
!conv2d_15/StatefulPartitionedCallІ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_236364conv2d_16_236366*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2349402#
!conv2d_16/StatefulPartitionedCallЉ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_236369batch_normalization_10_236371batch_normalization_10_236373batch_normalization_10_236375*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23573620
.batch_normalization_10/StatefulPartitionedCallГ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_236378conv2d_17_236380*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2351032#
!conv2d_17/StatefulPartitionedCallЉ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_236383batch_normalization_11_236385batch_normalization_11_236387batch_normalization_11_236389*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23582520
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2358672
add_5/PartitionedCallс
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_2358812
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_2352462,
*global_average_pooling2d_1/PartitionedCallх
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2358962
flatten_1/PartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_236396dense_3_236398*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2359312!
dense_3/StatefulPartitionedCallл
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2359642
dropout_2/PartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_236402dense_4_236404*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2360042!
dense_4/StatefulPartitionedCallл
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360372
dropout_3/PartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_236408dense_5_236410*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2360772!
dense_5/StatefulPartitionedCallО
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_236289*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЎ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_236291*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addС
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_236294*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addБ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_236296*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addС
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_236308*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addБ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_236310*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addС
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_236324*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addБ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_236326*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addС
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_236329*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addБ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_236331*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addС
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_236343*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addБ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_236345*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addС
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_236359*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addБ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_236361*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addС
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_236364*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addБ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_236366*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addС
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_236378*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addБ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_236380*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addД
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_236396*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЌ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_236398*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЕ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_236402* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЌ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_236404*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addД
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_236408*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЋ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_236410*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЧ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ќ
е
(__inference_model_1_layer_call_fn_237449
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2373502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
ж
d
H__inference_activation_4_layer_call_and_return_conditional_losses_239637

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs
Ў
Љ
6__inference_batch_normalization_8_layer_call_fn_239429

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2355072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
$
и
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238929

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ў
А
C__inference_model_1_layer_call_and_return_conditional_losses_236286
input_2
conv2d_9_235256
conv2d_9_235258
conv2d_10_235261
conv2d_10_235263 
batch_normalization_6_235341 
batch_normalization_6_235343 
batch_normalization_6_235345 
batch_normalization_6_235347
conv2d_11_235350
conv2d_11_235352 
batch_normalization_7_235430 
batch_normalization_7_235432 
batch_normalization_7_235434 
batch_normalization_7_235436
conv2d_12_235467
conv2d_12_235469
conv2d_13_235472
conv2d_13_235474 
batch_normalization_8_235552 
batch_normalization_8_235554 
batch_normalization_8_235556 
batch_normalization_8_235558
conv2d_14_235561
conv2d_14_235563 
batch_normalization_9_235641 
batch_normalization_9_235643 
batch_normalization_9_235645 
batch_normalization_9_235647
conv2d_15_235678
conv2d_15_235680
conv2d_16_235683
conv2d_16_235685!
batch_normalization_10_235763!
batch_normalization_10_235765!
batch_normalization_10_235767!
batch_normalization_10_235769
conv2d_17_235772
conv2d_17_235774!
batch_normalization_11_235852!
batch_normalization_11_235854!
batch_normalization_11_235856!
batch_normalization_11_235858
dense_3_235942
dense_3_235944
dense_4_236015
dense_4_236017
dense_5_236088
dense_5_236090
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallў
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_9_235256conv2d_9_235258*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2341722"
 conv2d_9/StatefulPartitionedCallЅ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_235261conv2d_10_235263*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2342102#
!conv2d_10/StatefulPartitionedCall 
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_235341batch_normalization_6_235343batch_normalization_6_235345batch_normalization_6_235347*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2352962/
-batch_normalization_6/StatefulPartitionedCallВ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_235350conv2d_11_235352*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2343732#
!conv2d_11/StatefulPartitionedCall 
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_235430batch_normalization_7_235432batch_normalization_7_235434batch_normalization_7_235436*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2353852/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_2354452
add_3/PartitionedCallс
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_2354592
activation_3/PartitionedCallЁ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_235467conv2d_12_235469*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2345372#
!conv2d_12/StatefulPartitionedCallІ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_235472conv2d_13_235474*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2345752#
!conv2d_13/StatefulPartitionedCall 
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_235552batch_normalization_8_235554batch_normalization_8_235556batch_normalization_8_235558*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2355072/
-batch_normalization_8/StatefulPartitionedCallВ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_235561conv2d_14_235563*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2347382#
!conv2d_14/StatefulPartitionedCall 
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_235641batch_normalization_9_235643batch_normalization_9_235645batch_normalization_9_235647*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2355962/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2356562
add_4/PartitionedCallс
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_2356702
activation_4/PartitionedCallЁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_235678conv2d_15_235680*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2349022#
!conv2d_15/StatefulPartitionedCallІ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_235683conv2d_16_235685*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2349402#
!conv2d_16/StatefulPartitionedCallЇ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_235763batch_normalization_10_235765batch_normalization_10_235767batch_normalization_10_235769*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23571820
.batch_normalization_10/StatefulPartitionedCallГ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_235772conv2d_17_235774*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2351032#
!conv2d_17/StatefulPartitionedCallЇ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_235852batch_normalization_11_235854batch_normalization_11_235856batch_normalization_11_235858*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23580720
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2358672
add_5/PartitionedCallс
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_2358812
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_2352462,
*global_average_pooling2d_1/PartitionedCallх
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2358962
flatten_1/PartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_235942dense_3_235944*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2359312!
dense_3/StatefulPartitionedCallѓ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2359592#
!dropout_2/StatefulPartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_236015dense_4_236017*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2360042!
dense_4/StatefulPartitionedCall
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360322#
!dropout_3/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_236088dense_5_236090*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2360772!
dense_5/StatefulPartitionedCallО
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_235256*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЎ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_235258*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addС
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_235261*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addБ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_235263*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addС
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_235350*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addБ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_235352*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addС
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_235467*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addБ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_235469*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addС
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_235472*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addБ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_235474*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addС
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_235561*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addБ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_235563*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addС
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_235678*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addБ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_235680*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addС
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_235683*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addБ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_235685*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addС
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_235772*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addБ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_235774*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addД
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_235942*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЌ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_235944*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЕ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_236015* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЌ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_236017*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addД
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_236088*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЋ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_236090*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/add
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
$
и
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_235596

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
љ
д
(__inference_model_1_layer_call_fn_238854

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:џџџџџџџџџ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2373502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
А
Љ
6__inference_batch_normalization_8_layer_call_fn_239442

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2355252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
с
~
)__inference_conv2d_9_layer_call_fn_234182

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2341722
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ж
d
H__inference_activation_3_layer_call_and_return_conditional_losses_235459

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs
$
и
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239576

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
$
и
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_235296

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у

*__inference_conv2d_13_layer_call_fn_234585

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2345752
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
џ
Ћ
C__inference_dense_4_layer_call_and_return_conditional_losses_236004

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluХ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addН
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч$
и
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_234832

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у

*__inference_conv2d_12_layer_call_fn_234547

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2345372
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ш$
й
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239895

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш$
й
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239717

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г 
­
E__inference_conv2d_16_layer_call_and_return_conditional_losses_234940

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
ReluЯ
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addР
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Щ

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239810

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@:::::W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 $
й
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_235718

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_240116

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у

*__inference_conv2d_17_layer_call_fn_235113

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2351032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
љ
q
__inference_loss_fn_14_240452?
;conv2d_16_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_16_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addg
IdentityIdentity$conv2d_16/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
А
Љ
6__inference_batch_normalization_7_layer_call_fn_239226

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2354032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239519

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_234863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

n
__inference_loss_fn_3_240309=
9conv2d_10_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_10_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/adde
IdentityIdentity"conv2d_10/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
у

*__inference_conv2d_10_layer_call_fn_234220

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2342102
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ђ
R
&__inference_add_4_layer_call_fn_239632
inputs_0
inputs_1
identityЕ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2356562
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22 :џџџџџџџџџ22 :Y U
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22 
"
_user_specified_name
inputs/1
я
W
;__inference_global_average_pooling2d_1_layer_call_fn_235252

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_2352462
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

c
*__inference_dropout_3_layer_call_fn_240200

inputs
identityЂStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
o
__inference_loss_fn_18_240504=
9dense_3_kernel_regularizer_square_readvariableop_resource
identityп
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/adde
IdentityIdentity"dense_3/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
љ
}
(__inference_dense_5_layer_call_fn_240257

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2360772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ь
m
__inference_loss_fn_21_240543;
7dense_4_bias_regularizer_square_readvariableop_resource
identityе
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_4_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addc
IdentityIdentity dense_4/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ь
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_235964

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш$
й
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_235034

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я
k
A__inference_add_5_layer_call_and_return_conditional_losses_235867

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22@:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs

o
__inference_loss_fn_11_240413=
9conv2d_14_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_14_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/adde
IdentityIdentity"conv2d_14/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ж
d
H__inference_activation_5_layer_call_and_return_conditional_losses_240031

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22@:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs


Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239125

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ћЉ
Ф
C__inference_model_1_layer_call_and_return_conditional_losses_238280

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityЂ:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpЂ:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpЂ<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpЂ9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpЂ;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpА
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOpО
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_9/Conv2DЇ
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOpЌ
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_9/BiasAddГ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOpд
conv2d_10/Conv2DConv2Dconv2d_9/BiasAdd:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_10/Conv2DЊ
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOpА
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/ReluЖ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOpМ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2(
&batch_normalization_6/FusedBatchNormV3
batch_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_6/Constђ
+batch_normalization_6/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_6/AssignMovingAvg/sub/x­
)batch_normalization_6/AssignMovingAvg/subSub4batch_normalization_6/AssignMovingAvg/sub/x:output:0$batch_normalization_6/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/subч
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_6/AssignMovingAvg/sub_1Sub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_6/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg/sub_1Е
)batch_normalization_6/AssignMovingAvg/mulMul/batch_normalization_6/AssignMovingAvg/sub_1:z:0-batch_normalization_6/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/mulс
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_6/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_6/AssignMovingAvg_1/sub/xЕ
+batch_normalization_6/AssignMovingAvg_1/subSub6batch_normalization_6/AssignMovingAvg_1/sub/x:output:0$batch_normalization_6/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subэ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_6/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2/
-batch_normalization_6/AssignMovingAvg_1/sub_1П
+batch_normalization_6/AssignMovingAvg_1/mulMul1batch_normalization_6/AssignMovingAvg_1/sub_1:z:0/batch_normalization_6/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/mulя
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOpх
conv2d_11/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_11/Conv2DЊ
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOpА
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_11/BiasAddЖ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOpМ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Я
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:2(
&batch_normalization_7/FusedBatchNormV3
batch_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_7/Constђ
+batch_normalization_7/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_7/AssignMovingAvg/sub/x­
)batch_normalization_7/AssignMovingAvg/subSub4batch_normalization_7/AssignMovingAvg/sub/x:output:0$batch_normalization_7/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/subч
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_7/AssignMovingAvg/sub_1Sub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_7/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg/sub_1Е
)batch_normalization_7/AssignMovingAvg/mulMul/batch_normalization_7/AssignMovingAvg/sub_1:z:0-batch_normalization_7/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mulс
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_7/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_7/AssignMovingAvg_1/sub/xЕ
+batch_normalization_7/AssignMovingAvg_1/subSub6batch_normalization_7/AssignMovingAvg_1/sub/x:output:0$batch_normalization_7/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subэ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_7/AssignMovingAvg_1/sub_1Sub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_7/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2/
-batch_normalization_7/AssignMovingAvg_1/sub_1П
+batch_normalization_7/AssignMovingAvg_1/mulMul1batch_normalization_7/AssignMovingAvg_1/sub_1:z:0/batch_normalization_7/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mulя
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp 
	add_3/addAddV2*batch_normalization_7/FusedBatchNormV3:y:0conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
	add_3/addw
activation_3/ReluReluadd_3/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
activation_3/ReluГ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpк
conv2d_12/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_12/Conv2DЊ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOpА
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/ReluГ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOpз
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_13/Conv2DЊ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOpА
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/ReluЖ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOpМ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1щ
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2(
&batch_normalization_8/FusedBatchNormV3
batch_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_8/Constђ
+batch_normalization_8/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_8/AssignMovingAvg/sub/x­
)batch_normalization_8/AssignMovingAvg/subSub4batch_normalization_8/AssignMovingAvg/sub/x:output:0$batch_normalization_8/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_8/AssignMovingAvg/subч
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_8/AssignMovingAvg/sub_1Sub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_8/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg/sub_1Е
)batch_normalization_8/AssignMovingAvg/mulMul/batch_normalization_8/AssignMovingAvg/sub_1:z:0-batch_normalization_8/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_8/AssignMovingAvg/mulс
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_8/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_8/AssignMovingAvg_1/sub/xЕ
+batch_normalization_8/AssignMovingAvg_1/subSub6batch_normalization_8/AssignMovingAvg_1/sub/x:output:0$batch_normalization_8/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg_1/subэ
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_8/AssignMovingAvg_1/sub_1Sub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_8/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_8/AssignMovingAvg_1/sub_1П
+batch_normalization_8/AssignMovingAvg_1/mulMul1batch_normalization_8/AssignMovingAvg_1/sub_1:z:0/batch_normalization_8/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_8/AssignMovingAvg_1/mulя
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_14/Conv2D/ReadVariableOpх
conv2d_14/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_14/Conv2DЊ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOpА
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_14/BiasAddЖ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOpМ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Я
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_14/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:2(
&batch_normalization_9/FusedBatchNormV3
batch_normalization_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_9/Constђ
+batch_normalization_9/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_9/AssignMovingAvg/sub/x­
)batch_normalization_9/AssignMovingAvg/subSub4batch_normalization_9/AssignMovingAvg/sub/x:output:0$batch_normalization_9/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/subч
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOpЬ
+batch_normalization_9/AssignMovingAvg/sub_1Sub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_9/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg/sub_1Е
)batch_normalization_9/AssignMovingAvg/mulMul/batch_normalization_9/AssignMovingAvg/sub_1:z:0-batch_normalization_9/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_9/AssignMovingAvg/mulс
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp6^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpј
-batch_normalization_9/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_9/AssignMovingAvg_1/sub/xЕ
+batch_normalization_9/AssignMovingAvg_1/subSub6batch_normalization_9/AssignMovingAvg_1/sub/x:output:0$batch_normalization_9/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/subэ
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpи
-batch_normalization_9/AssignMovingAvg_1/sub_1Sub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_9/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_9/AssignMovingAvg_1/sub_1П
+batch_normalization_9/AssignMovingAvg_1/mulMul1batch_normalization_9/AssignMovingAvg_1/sub_1:z:0/batch_normalization_9/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_9/AssignMovingAvg_1/mulя
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpЃ
	add_4/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0conv2d_12/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
	add_4/addw
activation_4/ReluReluadd_4/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
activation_4/ReluГ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOpк
conv2d_15/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_15/Conv2DЊ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpА
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/ReluГ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_16/Conv2D/ReadVariableOpз
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_16/Conv2DЊ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOpА
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/ReluЙ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOpП
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1ь
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1з
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2)
'batch_normalization_10/FusedBatchNormV3
batch_normalization_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_10/Constѕ
,batch_normalization_10/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_10/AssignMovingAvg/sub/xВ
*batch_normalization_10/AssignMovingAvg/subSub5batch_normalization_10/AssignMovingAvg/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_10/AssignMovingAvg/subъ
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOpб
,batch_normalization_10/AssignMovingAvg/sub_1Sub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_10/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_10/AssignMovingAvg/sub_1К
*batch_normalization_10/AssignMovingAvg/mulMul0batch_normalization_10/AssignMovingAvg/sub_1:z:0.batch_normalization_10/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_10/AssignMovingAvg/mulш
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_10/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_10/AssignMovingAvg_1/sub/xК
,batch_normalization_10/AssignMovingAvg_1/subSub7batch_normalization_10/AssignMovingAvg_1/sub/x:output:0%batch_normalization_10/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_10/AssignMovingAvg_1/sub№
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_10/AssignMovingAvg_1/sub_1Sub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_10/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_10/AssignMovingAvg_1/sub_1Ф
,batch_normalization_10/AssignMovingAvg_1/mulMul2batch_normalization_10/AssignMovingAvg_1/sub_1:z:00batch_normalization_10/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_10/AssignMovingAvg_1/mulі
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpГ
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_17/Conv2D/ReadVariableOpц
conv2d_17/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_17/Conv2DЊ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOpА
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_17/BiasAddЙ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOpП
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ь
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1е
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2)
'batch_normalization_11/FusedBatchNormV3
batch_normalization_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
batch_normalization_11/Constѕ
,batch_normalization_11/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,batch_normalization_11/AssignMovingAvg/sub/xВ
*batch_normalization_11/AssignMovingAvg/subSub5batch_normalization_11/AssignMovingAvg/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/subъ
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOpб
,batch_normalization_11/AssignMovingAvg/sub_1Sub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_11/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_11/AssignMovingAvg/sub_1К
*batch_normalization_11/AssignMovingAvg/mulMul0batch_normalization_11/AssignMovingAvg/sub_1:z:0.batch_normalization_11/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_11/AssignMovingAvg/mulш
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp7^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpћ
.batch_normalization_11/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.batch_normalization_11/AssignMovingAvg_1/sub/xК
,batch_normalization_11/AssignMovingAvg_1/subSub7batch_normalization_11/AssignMovingAvg_1/sub/x:output:0%batch_normalization_11/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/sub№
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpн
.batch_normalization_11/AssignMovingAvg_1/sub_1Sub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_11/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_11/AssignMovingAvg_1/sub_1Ф
,batch_normalization_11/AssignMovingAvg_1/mulMul2batch_normalization_11/AssignMovingAvg_1/sub_1:z:00batch_normalization_11/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_11/AssignMovingAvg_1/mulі
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpЄ
	add_5/addAddV2+batch_normalization_11/FusedBatchNormV3:y:0conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
	add_5/addw
activation_5/ReluReluadd_5/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
activation_5/ReluЗ
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesй
global_average_pooling2d_1/MeanMeanactivation_5/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling2d_1/Means
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
flatten_1/ConstЇ
flatten_1/ReshapeReshape(global_average_pooling2d_1/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten_1/ReshapeІ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЅ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЂ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_3/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/ConstІ
dropout_2/dropout/MulMuldense_3/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_2/dropout/Mul|
dropout_2/dropout/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeг
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yч
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_2/dropout/CastЃ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_2/dropout/Mul_1Ї
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_4/MatMul/ReadVariableOpЁ
dense_4/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЅ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЂ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/ConstІ
dropout_3/dropout/MulMuldense_4/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul|
dropout_3/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeг
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yч
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/CastЃ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul_1І
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/Sigmoidж
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЧ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addй
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addЪ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addй
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addЪ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addй
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addЪ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addй
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addЪ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addй
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addЪ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addй
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addЪ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addй
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addЪ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addй
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addЪ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addЬ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addХ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЭ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addХ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addЬ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addФ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЧ
IdentityIdentitydense_5/Sigmoid:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
А
Њ
7__inference_batch_normalization_11_layer_call_fn_240001

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2358072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч$
и
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_234669

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Щ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpа
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 $
й
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239792

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з
m
A__inference_add_5_layer_call_and_return_conditional_losses_240020
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:џџџџџџџџџ22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:џџџџџџџџџ22@:џџџџџџџџџ22@:Y U
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ22@
"
_user_specified_name
inputs/1
Ш

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239594

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А
Њ
7__inference_batch_normalization_10_layer_call_fn_239823

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2357182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
№
е
(__inference_model_1_layer_call_fn_237028
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identityЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2369292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ22
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
Ф
ф
C__inference_model_1_layer_call_and_return_conditional_losses_238652

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityА
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOpО
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_9/Conv2DЇ
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOpЌ
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_9/BiasAddГ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOpд
conv2d_10/Conv2DConv2Dconv2d_9/BiasAdd:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_10/Conv2DЊ
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOpА
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_10/ReluЖ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOpМ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3Г
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOpх
conv2d_11/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22*
paddingSAME*
strides
2
conv2d_11/Conv2DЊ
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOpА
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ222
conv2d_11/BiasAddЖ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOpМ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3 
	add_3/addAddV2*batch_normalization_7/FusedBatchNormV3:y:0conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ222
	add_3/addw
activation_3/ReluReluadd_3/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ222
activation_3/ReluГ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpк
conv2d_12/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_12/Conv2DЊ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOpА
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_12/ReluГ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_13/Conv2D/ReadVariableOpз
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_13/Conv2DЊ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOpА
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_13/ReluЖ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOpМ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1щ
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3Г
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_14/Conv2D/ReadVariableOpх
conv2d_14/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 *
paddingSAME*
strides
2
conv2d_14/Conv2DЊ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOpА
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
conv2d_14/BiasAddЖ
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOpМ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1щ
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_14/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3Ѓ
	add_4/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0conv2d_12/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
	add_4/addw
activation_4/ReluReluadd_4/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2
activation_4/ReluГ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOpк
conv2d_15/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_15/Conv2DЊ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpА
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_15/ReluГ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_16/Conv2D/ReadVariableOpз
conv2d_16/Conv2DConv2Dconv2d_15/Relu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_16/Conv2DЊ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOpА
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_16/ReluЙ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_10/ReadVariableOpП
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_10/ReadVariableOp_1ь
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ъ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3Г
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_17/Conv2D/ReadVariableOpц
conv2d_17/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@*
paddingSAME*
strides
2
conv2d_17/Conv2DЊ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOpА
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
conv2d_17/BiasAddЙ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOpП
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ь
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpђ
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ш
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3Є
	add_5/addAddV2+batch_normalization_11/FusedBatchNormV3:y:0conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
	add_5/addw
activation_5/ReluReluadd_5/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ22@2
activation_5/ReluЗ
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesй
global_average_pooling2d_1/MeanMeanactivation_5/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling2d_1/Means
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   2
flatten_1/ConstЇ
flatten_1/ReshapeReshape(global_average_pooling2d_1/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
flatten_1/ReshapeІ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЅ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЂ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_3/Relu
dropout_2/IdentityIdentitydense_3/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_2/IdentityЇ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_4/MatMul/ReadVariableOpЁ
dense_4/MatMulMatMuldropout_2/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЅ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЂ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/Relu
dropout_3/IdentityIdentitydense_4/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_3/IdentityІ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldropout_3/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/Sigmoidж
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЧ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addй
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addЪ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addй
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addЪ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addй
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addЪ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addй
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addЪ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addй
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addЪ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addй
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addЪ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addй
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addЪ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addй
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addЪ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addЬ
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addХ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЭ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addХ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addЬ
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addФ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addg
IdentityIdentitydense_5/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22:::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 
Г 
­
E__inference_conv2d_10_layer_call_and_return_conditional_losses_234210

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluЯ
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addР
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
њ
Њ
7__inference_batch_normalization_11_layer_call_fn_239939

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2352282
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
џ
Ћ
C__inference_dense_4_layer_call_and_return_conditional_losses_240169

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluХ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addН
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ј
p
__inference_loss_fn_8_240374?
;conv2d_13_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_13_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addg
IdentityIdentity$conv2d_13/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ш

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_235614

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22 :::::W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 $
й
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239970

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22@:@:@:@:@:*
epsilon%o:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
ConstА
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xП
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subЅ
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpо
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1Ч
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mulЧ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЖ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЧ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЋ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1б
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mulе
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpО
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:џџџџџџџџџ22@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_235403

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ22:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ22:::::W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

o
__inference_loss_fn_13_240439=
9conv2d_15_bias_regularizer_square_readvariableop_resource
identityк
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_15_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/adde
IdentityIdentity"conv2d_15/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 

ч
C__inference_model_1_layer_call_and_return_conditional_losses_237350

inputs
conv2d_9_237033
conv2d_9_237035
conv2d_10_237038
conv2d_10_237040 
batch_normalization_6_237043 
batch_normalization_6_237045 
batch_normalization_6_237047 
batch_normalization_6_237049
conv2d_11_237052
conv2d_11_237054 
batch_normalization_7_237057 
batch_normalization_7_237059 
batch_normalization_7_237061 
batch_normalization_7_237063
conv2d_12_237068
conv2d_12_237070
conv2d_13_237073
conv2d_13_237075 
batch_normalization_8_237078 
batch_normalization_8_237080 
batch_normalization_8_237082 
batch_normalization_8_237084
conv2d_14_237087
conv2d_14_237089 
batch_normalization_9_237092 
batch_normalization_9_237094 
batch_normalization_9_237096 
batch_normalization_9_237098
conv2d_15_237103
conv2d_15_237105
conv2d_16_237108
conv2d_16_237110!
batch_normalization_10_237113!
batch_normalization_10_237115!
batch_normalization_10_237117!
batch_normalization_10_237119
conv2d_17_237122
conv2d_17_237124!
batch_normalization_11_237127!
batch_normalization_11_237129!
batch_normalization_11_237131!
batch_normalization_11_237133
dense_3_237140
dense_3_237142
dense_4_237146
dense_4_237148
dense_5_237152
dense_5_237154
identityЂ.batch_normalization_10/StatefulPartitionedCallЂ.batch_normalization_11/StatefulPartitionedCallЂ-batch_normalization_6/StatefulPartitionedCallЂ-batch_normalization_7/StatefulPartitionedCallЂ-batch_normalization_8/StatefulPartitionedCallЂ-batch_normalization_9/StatefulPartitionedCallЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ!conv2d_15/StatefulPartitionedCallЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ conv2d_9/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCall§
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_237033conv2d_9_237035*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_2341722"
 conv2d_9/StatefulPartitionedCallЅ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_237038conv2d_10_237040*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_2342102#
!conv2d_10/StatefulPartitionedCallЂ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_6_237043batch_normalization_6_237045batch_normalization_6_237047batch_normalization_6_237049*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2353142/
-batch_normalization_6/StatefulPartitionedCallВ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_11_237052conv2d_11_237054*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_2343732#
!conv2d_11/StatefulPartitionedCallЂ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_7_237057batch_normalization_7_237059batch_normalization_7_237061batch_normalization_7_237063*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2354032/
-batch_normalization_7/StatefulPartitionedCall
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_2354452
add_3/PartitionedCallс
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_2354592
activation_3/PartitionedCallЁ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_12_237068conv2d_12_237070*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_2345372#
!conv2d_12/StatefulPartitionedCallІ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_237073conv2d_13_237075*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_2345752#
!conv2d_13/StatefulPartitionedCallЂ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_8_237078batch_normalization_8_237080batch_normalization_8_237082batch_normalization_8_237084*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2355252/
-batch_normalization_8/StatefulPartitionedCallВ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_14_237087conv2d_14_237089*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_2347382#
!conv2d_14/StatefulPartitionedCallЂ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_9_237092batch_normalization_9_237094batch_normalization_9_237096batch_normalization_9_237098*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2356142/
-batch_normalization_9/StatefulPartitionedCall
add_4/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_2356562
add_4/PartitionedCallс
activation_4/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_2356702
activation_4/PartitionedCallЁ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_15_237103conv2d_15_237105*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2349022#
!conv2d_15/StatefulPartitionedCallІ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0conv2d_16_237108conv2d_16_237110*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2349402#
!conv2d_16/StatefulPartitionedCallЉ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_10_237113batch_normalization_10_237115batch_normalization_10_237117batch_normalization_10_237119*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23573620
.batch_normalization_10/StatefulPartitionedCallГ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_17_237122conv2d_17_237124*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2351032#
!conv2d_17/StatefulPartitionedCallЉ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_11_237127batch_normalization_11_237129batch_normalization_11_237131batch_normalization_11_237133*
Tin	
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23582520
.batch_normalization_11/StatefulPartitionedCall
add_5/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_5_layer_call_and_return_conditional_losses_2358672
add_5/PartitionedCallс
activation_5/PartitionedCallPartitionedCalladd_5/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_2358812
activation_5/PartitionedCall
*global_average_pooling2d_1/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_2352462,
*global_average_pooling2d_1/PartitionedCallх
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2358962
flatten_1/PartitionedCall
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_237140dense_3_237142*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2359312!
dense_3/StatefulPartitionedCallл
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2359642
dropout_2/PartitionedCall
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_237146dense_4_237148*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2360042!
dense_4/StatefulPartitionedCallл
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2360372
dropout_3/PartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_237152dense_5_237154*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2360772!
dense_5/StatefulPartitionedCallО
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_237033*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addЎ
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_237035*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/addС
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_237038*&
_output_shapes
:*
dtype024
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_10/kernel/Regularizer/SquareЁ
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_10/kernel/Regularizer/ConstТ
 conv2d_10/kernel/Regularizer/SumSum'conv2d_10/kernel/Regularizer/Square:y:0+conv2d_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/Sum
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_10/kernel/Regularizer/mul/xФ
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/mul
"conv2d_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_10/kernel/Regularizer/add/xС
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/add/x:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_10/kernel/Regularizer/addБ
0conv2d_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_237040*
_output_shapes
:*
dtype022
0conv2d_10/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_10/bias/Regularizer/SquareSquare8conv2d_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_10/bias/Regularizer/Square
 conv2d_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_10/bias/Regularizer/ConstК
conv2d_10/bias/Regularizer/SumSum%conv2d_10/bias/Regularizer/Square:y:0)conv2d_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/Sum
 conv2d_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_10/bias/Regularizer/mul/xМ
conv2d_10/bias/Regularizer/mulMul)conv2d_10/bias/Regularizer/mul/x:output:0'conv2d_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/mul
 conv2d_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_10/bias/Regularizer/add/xЙ
conv2d_10/bias/Regularizer/addAddV2)conv2d_10/bias/Regularizer/add/x:output:0"conv2d_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_10/bias/Regularizer/addС
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_237052*&
_output_shapes
:*
dtype024
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_11/kernel/Regularizer/SquareЁ
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_11/kernel/Regularizer/ConstТ
 conv2d_11/kernel/Regularizer/SumSum'conv2d_11/kernel/Regularizer/Square:y:0+conv2d_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/Sum
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_11/kernel/Regularizer/mul/xФ
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/mul
"conv2d_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_11/kernel/Regularizer/add/xС
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/add/x:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_11/kernel/Regularizer/addБ
0conv2d_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_237054*
_output_shapes
:*
dtype022
0conv2d_11/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_11/bias/Regularizer/SquareSquare8conv2d_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_11/bias/Regularizer/Square
 conv2d_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_11/bias/Regularizer/ConstК
conv2d_11/bias/Regularizer/SumSum%conv2d_11/bias/Regularizer/Square:y:0)conv2d_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/Sum
 conv2d_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_11/bias/Regularizer/mul/xМ
conv2d_11/bias/Regularizer/mulMul)conv2d_11/bias/Regularizer/mul/x:output:0'conv2d_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/mul
 conv2d_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_11/bias/Regularizer/add/xЙ
conv2d_11/bias/Regularizer/addAddV2)conv2d_11/bias/Regularizer/add/x:output:0"conv2d_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_11/bias/Regularizer/addС
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_237068*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareЁ
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/ConstТ
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/Sum
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_12/kernel/Regularizer/mul/xФ
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul
"conv2d_12/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_12/kernel/Regularizer/add/xС
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/add/x:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/addБ
0conv2d_12/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_237070*
_output_shapes
: *
dtype022
0conv2d_12/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_12/bias/Regularizer/SquareSquare8conv2d_12/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_12/bias/Regularizer/Square
 conv2d_12/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_12/bias/Regularizer/ConstК
conv2d_12/bias/Regularizer/SumSum%conv2d_12/bias/Regularizer/Square:y:0)conv2d_12/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/Sum
 conv2d_12/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_12/bias/Regularizer/mul/xМ
conv2d_12/bias/Regularizer/mulMul)conv2d_12/bias/Regularizer/mul/x:output:0'conv2d_12/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/mul
 conv2d_12/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_12/bias/Regularizer/add/xЙ
conv2d_12/bias/Regularizer/addAddV2)conv2d_12/bias/Regularizer/add/x:output:0"conv2d_12/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_12/bias/Regularizer/addС
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_237073*&
_output_shapes
:  *
dtype024
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_13/kernel/Regularizer/SquareЁ
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_13/kernel/Regularizer/ConstТ
 conv2d_13/kernel/Regularizer/SumSum'conv2d_13/kernel/Regularizer/Square:y:0+conv2d_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/Sum
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_13/kernel/Regularizer/mul/xФ
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/mul
"conv2d_13/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_13/kernel/Regularizer/add/xС
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/add/x:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_13/kernel/Regularizer/addБ
0conv2d_13/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_237075*
_output_shapes
: *
dtype022
0conv2d_13/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_13/bias/Regularizer/SquareSquare8conv2d_13/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_13/bias/Regularizer/Square
 conv2d_13/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_13/bias/Regularizer/ConstК
conv2d_13/bias/Regularizer/SumSum%conv2d_13/bias/Regularizer/Square:y:0)conv2d_13/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/Sum
 conv2d_13/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_13/bias/Regularizer/mul/xМ
conv2d_13/bias/Regularizer/mulMul)conv2d_13/bias/Regularizer/mul/x:output:0'conv2d_13/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/mul
 conv2d_13/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_13/bias/Regularizer/add/xЙ
conv2d_13/bias/Regularizer/addAddV2)conv2d_13/bias/Regularizer/add/x:output:0"conv2d_13/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_13/bias/Regularizer/addС
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_237087*&
_output_shapes
:  *
dtype024
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_14/kernel/Regularizer/SquareЁ
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_14/kernel/Regularizer/ConstТ
 conv2d_14/kernel/Regularizer/SumSum'conv2d_14/kernel/Regularizer/Square:y:0+conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/Sum
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_14/kernel/Regularizer/mul/xФ
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/mul
"conv2d_14/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_14/kernel/Regularizer/add/xС
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/add/x:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_14/kernel/Regularizer/addБ
0conv2d_14/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_237089*
_output_shapes
: *
dtype022
0conv2d_14/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_14/bias/Regularizer/SquareSquare8conv2d_14/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_14/bias/Regularizer/Square
 conv2d_14/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_14/bias/Regularizer/ConstК
conv2d_14/bias/Regularizer/SumSum%conv2d_14/bias/Regularizer/Square:y:0)conv2d_14/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/Sum
 conv2d_14/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_14/bias/Regularizer/mul/xМ
conv2d_14/bias/Regularizer/mulMul)conv2d_14/bias/Regularizer/mul/x:output:0'conv2d_14/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/mul
 conv2d_14/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_14/bias/Regularizer/add/xЙ
conv2d_14/bias/Regularizer/addAddV2)conv2d_14/bias/Regularizer/add/x:output:0"conv2d_14/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_14/bias/Regularizer/addС
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_237103*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addБ
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_237105*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/addС
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_237108*&
_output_shapes
:@@*
dtype024
2conv2d_16/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_16/kernel/Regularizer/SquareSquare:conv2d_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_16/kernel/Regularizer/SquareЁ
"conv2d_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_16/kernel/Regularizer/ConstТ
 conv2d_16/kernel/Regularizer/SumSum'conv2d_16/kernel/Regularizer/Square:y:0+conv2d_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/Sum
"conv2d_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_16/kernel/Regularizer/mul/xФ
 conv2d_16/kernel/Regularizer/mulMul+conv2d_16/kernel/Regularizer/mul/x:output:0)conv2d_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/mul
"conv2d_16/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_16/kernel/Regularizer/add/xС
 conv2d_16/kernel/Regularizer/addAddV2+conv2d_16/kernel/Regularizer/add/x:output:0$conv2d_16/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_16/kernel/Regularizer/addБ
0conv2d_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_16_237110*
_output_shapes
:@*
dtype022
0conv2d_16/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_16/bias/Regularizer/SquareSquare8conv2d_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_16/bias/Regularizer/Square
 conv2d_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_16/bias/Regularizer/ConstК
conv2d_16/bias/Regularizer/SumSum%conv2d_16/bias/Regularizer/Square:y:0)conv2d_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/Sum
 conv2d_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_16/bias/Regularizer/mul/xМ
conv2d_16/bias/Regularizer/mulMul)conv2d_16/bias/Regularizer/mul/x:output:0'conv2d_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/mul
 conv2d_16/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_16/bias/Regularizer/add/xЙ
conv2d_16/bias/Regularizer/addAddV2)conv2d_16/bias/Regularizer/add/x:output:0"conv2d_16/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_16/bias/Regularizer/addС
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_237122*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addБ
0conv2d_17/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_17_237124*
_output_shapes
:@*
dtype022
0conv2d_17/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_17/bias/Regularizer/SquareSquare8conv2d_17/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_17/bias/Regularizer/Square
 conv2d_17/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_17/bias/Regularizer/ConstК
conv2d_17/bias/Regularizer/SumSum%conv2d_17/bias/Regularizer/Square:y:0)conv2d_17/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/Sum
 conv2d_17/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_17/bias/Regularizer/mul/xМ
conv2d_17/bias/Regularizer/mulMul)conv2d_17/bias/Regularizer/mul/x:output:0'conv2d_17/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/mul
 conv2d_17/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_17/bias/Regularizer/add/xЙ
conv2d_17/bias/Regularizer/addAddV2)conv2d_17/bias/Regularizer/add/x:output:0"conv2d_17/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_17/bias/Regularizer/addД
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_237140*
_output_shapes
:	@*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOpД
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_3/kernel/Regularizer/Square
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/ConstК
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_3/kernel/Regularizer/mul/xМ
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul
 dense_3/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_3/kernel/Regularizer/add/xЙ
dense_3/kernel/Regularizer/addAddV2)dense_3/kernel/Regularizer/add/x:output:0"dense_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/addЌ
.dense_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_237142*
_output_shapes	
:*
dtype020
.dense_3/bias/Regularizer/Square/ReadVariableOpЊ
dense_3/bias/Regularizer/SquareSquare6dense_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_3/bias/Regularizer/Square
dense_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_3/bias/Regularizer/ConstВ
dense_3/bias/Regularizer/SumSum#dense_3/bias/Regularizer/Square:y:0'dense_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/Sum
dense_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_3/bias/Regularizer/mul/xД
dense_3/bias/Regularizer/mulMul'dense_3/bias/Regularizer/mul/x:output:0%dense_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/mul
dense_3/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_3/bias/Regularizer/add/xБ
dense_3/bias/Regularizer/addAddV2'dense_3/bias/Regularizer/add/x:output:0 dense_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_3/bias/Regularizer/addЕ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_237146* 
_output_shapes
:
*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOpЕ
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/ConstК
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_4/kernel/Regularizer/mul/xМ
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul
 dense_4/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_4/kernel/Regularizer/add/xЙ
dense_4/kernel/Regularizer/addAddV2)dense_4/kernel/Regularizer/add/x:output:0"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/addЌ
.dense_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_237148*
_output_shapes	
:*
dtype020
.dense_4/bias/Regularizer/Square/ReadVariableOpЊ
dense_4/bias/Regularizer/SquareSquare6dense_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2!
dense_4/bias/Regularizer/Square
dense_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_4/bias/Regularizer/ConstВ
dense_4/bias/Regularizer/SumSum#dense_4/bias/Regularizer/Square:y:0'dense_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/Sum
dense_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_4/bias/Regularizer/mul/xД
dense_4/bias/Regularizer/mulMul'dense_4/bias/Regularizer/mul/x:output:0%dense_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/mul
dense_4/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_4/bias/Regularizer/add/xБ
dense_4/bias/Regularizer/addAddV2'dense_4/bias/Regularizer/add/x:output:0 dense_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_4/bias/Regularizer/addД
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_237152*
_output_shapes
:	*
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOpД
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!dense_5/kernel/Regularizer/Square
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/ConstК
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_5/kernel/Regularizer/mul/xМ
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul
 dense_5/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/add/xЙ
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/add/x:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/addЋ
.dense_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_237154*
_output_shapes
:*
dtype020
.dense_5/bias/Regularizer/Square/ReadVariableOpЉ
dense_5/bias/Regularizer/SquareSquare6dense_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2!
dense_5/bias/Regularizer/Square
dense_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_5/bias/Regularizer/ConstВ
dense_5/bias/Regularizer/SumSum#dense_5/bias/Regularizer/Square:y:0'dense_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/Sum
dense_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense_5/bias/Regularizer/mul/xД
dense_5/bias/Regularizer/mulMul'dense_5/bias/Regularizer/mul/x:output:0%dense_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/mul
dense_5/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_5/bias/Regularizer/add/xБ
dense_5/bias/Regularizer/addAddV2'dense_5/bias/Regularizer/add/x:output:0 dense_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_5/bias/Regularizer/addЧ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*№
_input_shapesо
л:џџџџџџџџџ22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ22
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: 

I
-__inference_activation_4_layer_call_fn_239642

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_2356702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22 :W S
/
_output_shapes
:џџџџџџџџџ22 
 
_user_specified_nameinputs

Ќ
D__inference_conv2d_9_layer_call_and_return_conditional_losses_234172

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddЭ
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_9/kernel/Regularizer/Square
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_9/kernel/Regularizer/ConstО
conv2d_9/kernel/Regularizer/SumSum&conv2d_9/kernel/Regularizer/Square:y:0*conv2d_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/Sum
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_9/kernel/Regularizer/mul/xР
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/mul
!conv2d_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_9/kernel/Regularizer/add/xН
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/add/x:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_9/kernel/Regularizer/addО
/conv2d_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/conv2d_9/bias/Regularizer/Square/ReadVariableOpЌ
 conv2d_9/bias/Regularizer/SquareSquare7conv2d_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 conv2d_9/bias/Regularizer/Square
conv2d_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
conv2d_9/bias/Regularizer/ConstЖ
conv2d_9/bias/Regularizer/SumSum$conv2d_9/bias/Regularizer/Square:y:0(conv2d_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/Sum
conv2d_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d_9/bias/Regularizer/mul/xИ
conv2d_9/bias/Regularizer/mulMul(conv2d_9/bias/Regularizer/mul/x:output:0&conv2d_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/mul
conv2d_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d_9/bias/Regularizer/add/xЕ
conv2d_9/bias/Regularizer/addAddV2(conv2d_9/bias/Regularizer/add/x:output:0!conv2d_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d_9/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
љ
q
__inference_loss_fn_16_240478?
;conv2d_17_kernel_regularizer_square_readvariableop_resource
identityь
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_17_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_17/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_17/kernel/Regularizer/SquareSquare:conv2d_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_17/kernel/Regularizer/SquareЁ
"conv2d_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_17/kernel/Regularizer/ConstТ
 conv2d_17/kernel/Regularizer/SumSum'conv2d_17/kernel/Regularizer/Square:y:0+conv2d_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/Sum
"conv2d_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_17/kernel/Regularizer/mul/xФ
 conv2d_17/kernel/Regularizer/mulMul+conv2d_17/kernel/Regularizer/mul/x:output:0)conv2d_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/mul
"conv2d_17/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_17/kernel/Regularizer/add/xС
 conv2d_17/kernel/Regularizer/addAddV2+conv2d_17/kernel/Regularizer/add/x:output:0$conv2d_17/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_17/kernel/Regularizer/addg
IdentityIdentity$conv2d_17/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
j
і
__inference__traced_save_240740
file_prefix.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fce83d12a6c048fa822b0819b35ab498/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameС
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*г
valueЩBЦ0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesш
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *>
dtypes4
2202
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*З
_input_shapesЅ
Ђ: ::::::::::::::: : :  : : : : : :  : : : : : : @:@:@@:@:@:@:@:@:@@:@:@:@:@:@:	@::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:,%(
&
_output_shapes
:@@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:%+!

_output_shapes
:	@:!,

_output_shapes	
::&-"
 
_output_shapes
:
:!.

_output_shapes	
::%/!

_output_shapes
:	: 0

_output_shapes
::1

_output_shapes
: 
Г 
­
E__inference_conv2d_15_layer_call_and_return_conditional_losses_234902

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
ReluЯ
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpС
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_15/kernel/Regularizer/SquareЁ
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_15/kernel/Regularizer/ConstТ
 conv2d_15/kernel/Regularizer/SumSum'conv2d_15/kernel/Regularizer/Square:y:0+conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/Sum
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"conv2d_15/kernel/Regularizer/mul/xФ
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/mul
"conv2d_15/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_15/kernel/Regularizer/add/xС
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/add/x:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_15/kernel/Regularizer/addР
0conv2d_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_15/bias/Regularizer/Square/ReadVariableOpЏ
!conv2d_15/bias/Regularizer/SquareSquare8conv2d_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_15/bias/Regularizer/Square
 conv2d_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_15/bias/Regularizer/ConstК
conv2d_15/bias/Regularizer/SumSum%conv2d_15/bias/Regularizer/Square:y:0)conv2d_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/Sum
 conv2d_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 conv2d_15/bias/Regularizer/mul/xМ
conv2d_15/bias/Regularizer/mulMul)conv2d_15/bias/Regularizer/mul/x:output:0'conv2d_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/mul
 conv2d_15/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_15/bias/Regularizer/add/xЙ
conv2d_15/bias/Regularizer/addAddV2)conv2d_15/bias/Regularizer/add/x:output:0"conv2d_15/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_15/bias/Regularizer/add
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_default
C
input_28
serving_default_input_2:0џџџџџџџџџ22;
dense_50
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Лц
А
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer-27
layer_with_weights-17
layer-28
	variables
regularization_losses
 trainable_variables
!	keras_api
"
signatures
+к&call_and_return_all_conditional_losses
л__call__
м_default_save_signature"ќ
_tf_keras_modelэћ{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}], ["conv2d_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}], ["conv2d_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}], ["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["global_average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}], ["conv2d_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "name": "add_4", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}], ["conv2d_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["add_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "name": "add_5", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}], ["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["add_5", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_1", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["global_average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
Ю


#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Ї	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}}
а


)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+п&call_and_return_all_conditional_losses
р__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+с&call_and_return_all_conditional_losses
т__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
в


8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}

G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 16]}, {"class_name": "TensorShape", "items": [null, 50, 50, 16]}]}
Д
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
а


Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
а


Ukernel
Vbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
	
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+я&call_and_return_all_conditional_losses
№__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
в


dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+ё&call_and_return_all_conditional_losses
ђ__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
	
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
+ѓ&call_and_return_all_conditional_losses
є__call__"У
_tf_keras_layerЉ{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}

s	variables
tregularization_losses
utrainable_variables
v	keras_api
+ѕ&call_and_return_all_conditional_losses
і__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 32]}, {"class_name": "TensorShape", "items": [null, 50, 50, 32]}]}
Д
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+ї&call_and_return_all_conditional_losses
ј__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
б


{kernel
|bias
}	variables
~regularization_losses
trainable_variables
	keras_api
+љ&call_and_return_all_conditional_losses
њ__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
ж

kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+ћ&call_and_return_all_conditional_losses
ќ__call__"Љ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Є	
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
+§&call_and_return_all_conditional_losses
ў__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
и

kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+џ&call_and_return_all_conditional_losses
__call__"Ћ	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Є	
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Х
_tf_keras_layerЋ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}

	variables
 regularization_losses
Ёtrainable_variables
Ђ	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerэ{"class_name": "Add", "name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 64]}, {"class_name": "TensorShape", "items": [null, 50, 50, 64]}]}
И
Ѓ	variables
Єregularization_losses
Ѕtrainable_variables
І	keras_api
+&call_and_return_all_conditional_losses
__call__"Ѓ
_tf_keras_layer{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
њ
Ї	variables
Јregularization_losses
Љtrainable_variables
Њ	keras_api
+&call_and_return_all_conditional_losses
__call__"х
_tf_keras_layerЫ{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Щ
Ћ	variables
Ќregularization_losses
­trainable_variables
Ў	keras_api
+&call_and_return_all_conditional_losses
__call__"Д
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
р
Џkernel
	Аbias
Б	variables
Вregularization_losses
Гtrainable_variables
Д	keras_api
+&call_and_return_all_conditional_losses
__call__"Г
_tf_keras_layer{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
Ш
Е	variables
Жregularization_losses
Зtrainable_variables
И	keras_api
+&call_and_return_all_conditional_losses
__call__"Г
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
т
Йkernel
	Кbias
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
+&call_and_return_all_conditional_losses
__call__"Е
_tf_keras_layer{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Ш
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
+&call_and_return_all_conditional_losses
__call__"Г
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
у
Уkernel
	Фbias
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
+&call_and_return_all_conditional_losses
__call__"Ж
_tf_keras_layer{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Ј
#0
$1
)2
*3
04
15
26
37
88
99
?10
@11
A12
B13
O14
P15
U16
V17
\18
]19
^20
_21
d22
e23
k24
l25
m26
n27
{28
|29
30
31
32
33
34
35
36
37
38
39
40
41
Џ42
А43
Й44
К45
У46
Ф47"
trackable_list_wrapper
ю
0
1
2
3
4
5
6
7
8
9
10
 11
Ё12
Ђ13
Ѓ14
Є15
Ѕ16
І17
Ї18
Ј19
Љ20
Њ21
Ћ22
Ќ23"
trackable_list_wrapper
Ф
#0
$1
)2
*3
04
15
86
97
?8
@9
O10
P11
U12
V13
\14
]15
d16
e17
k18
l19
{20
|21
22
23
24
25
26
27
28
29
Џ30
А31
Й32
К33
У34
Ф35"
trackable_list_wrapper
г
	variables
regularization_losses
Щnon_trainable_variables
 trainable_variables
Ъlayer_metrics
 Ыlayer_regularization_losses
Ьlayers
Эmetrics
л__call__
м_default_save_signature
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
-
­serving_default"
signature_map
):'2conv2d_9/kernel
:2conv2d_9/bias
.
#0
$1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
Е
%	variables
&regularization_losses
Юnon_trainable_variables
'trainable_variables
Яlayer_metrics
 аlayer_regularization_losses
бlayers
вmetrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_10/kernel
:2conv2d_10/bias
.
)0
*1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
Е
+	variables
,regularization_losses
гnon_trainable_variables
-trainable_variables
дlayer_metrics
 еlayer_regularization_losses
жlayers
зmetrics
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
Е
4	variables
5regularization_losses
иnon_trainable_variables
6trainable_variables
йlayer_metrics
 кlayer_regularization_losses
лlayers
мmetrics
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_11/kernel
:2conv2d_11/bias
.
80
91"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
Е
:	variables
;regularization_losses
нnon_trainable_variables
<trainable_variables
оlayer_metrics
 пlayer_regularization_losses
рlayers
сmetrics
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
Е
C	variables
Dregularization_losses
тnon_trainable_variables
Etrainable_variables
уlayer_metrics
 фlayer_regularization_losses
хlayers
цmetrics
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
G	variables
Hregularization_losses
чnon_trainable_variables
Itrainable_variables
шlayer_metrics
 щlayer_regularization_losses
ъlayers
ыmetrics
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
K	variables
Lregularization_losses
ьnon_trainable_variables
Mtrainable_variables
эlayer_metrics
 юlayer_regularization_losses
яlayers
№metrics
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
.
O0
P1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
Е
Q	variables
Rregularization_losses
ёnon_trainable_variables
Strainable_variables
ђlayer_metrics
 ѓlayer_regularization_losses
єlayers
ѕmetrics
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_13/kernel
: 2conv2d_13/bias
.
U0
V1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
Е
W	variables
Xregularization_losses
іnon_trainable_variables
Ytrainable_variables
їlayer_metrics
 јlayer_regularization_losses
љlayers
њmetrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
<
\0
]1
^2
_3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
Е
`	variables
aregularization_losses
ћnon_trainable_variables
btrainable_variables
ќlayer_metrics
 §layer_regularization_losses
ўlayers
џmetrics
№__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_14/kernel
: 2conv2d_14/bias
.
d0
e1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
Е
f	variables
gregularization_losses
non_trainable_variables
htrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
ђ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
Е
o	variables
pregularization_losses
non_trainable_variables
qtrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
є__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
s	variables
tregularization_losses
non_trainable_variables
utrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
і__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
w	variables
xregularization_losses
non_trainable_variables
ytrainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
ј__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_15/kernel
:@2conv2d_15/bias
.
{0
|1"
trackable_list_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
Е
}	variables
~regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
њ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_16/kernel
:@2conv2d_16/bias
0
0
1"
trackable_list_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
	variables
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
 layer_regularization_losses
layers
metrics
ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
	variables
regularization_losses
non_trainable_variables
trainable_variables
layer_metrics
  layer_regularization_losses
Ёlayers
Ђmetrics
ў__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_17/kernel
:@2conv2d_17/bias
0
0
1"
trackable_list_wrapper
0
Ѕ0
І1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
	variables
regularization_losses
Ѓnon_trainable_variables
trainable_variables
Єlayer_metrics
 Ѕlayer_regularization_losses
Іlayers
Їmetrics
__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_11/gamma
):'@2batch_normalization_11/beta
2:0@ (2"batch_normalization_11/moving_mean
6:4@ (2&batch_normalization_11/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
	variables
regularization_losses
Јnon_trainable_variables
trainable_variables
Љlayer_metrics
 Њlayer_regularization_losses
Ћlayers
Ќmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
	variables
 regularization_losses
­non_trainable_variables
Ёtrainable_variables
Ўlayer_metrics
 Џlayer_regularization_losses
Аlayers
Бmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѓ	variables
Єregularization_losses
Вnon_trainable_variables
Ѕtrainable_variables
Гlayer_metrics
 Дlayer_regularization_losses
Еlayers
Жmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ї	variables
Јregularization_losses
Зnon_trainable_variables
Љtrainable_variables
Иlayer_metrics
 Йlayer_regularization_losses
Кlayers
Лmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћ	variables
Ќregularization_losses
Мnon_trainable_variables
­trainable_variables
Нlayer_metrics
 Оlayer_regularization_losses
Пlayers
Рmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_3/kernel
:2dense_3/bias
0
Џ0
А1"
trackable_list_wrapper
0
Ї0
Ј1"
trackable_list_wrapper
0
Џ0
А1"
trackable_list_wrapper
И
Б	variables
Вregularization_losses
Сnon_trainable_variables
Гtrainable_variables
Тlayer_metrics
 Уlayer_regularization_losses
Фlayers
Хmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Е	variables
Жregularization_losses
Цnon_trainable_variables
Зtrainable_variables
Чlayer_metrics
 Шlayer_regularization_losses
Щlayers
Ъmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_4/kernel
:2dense_4/bias
0
Й0
К1"
trackable_list_wrapper
0
Љ0
Њ1"
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
И
Л	variables
Мregularization_losses
Ыnon_trainable_variables
Нtrainable_variables
Ьlayer_metrics
 Эlayer_regularization_losses
Юlayers
Яmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
П	variables
Рregularization_losses
аnon_trainable_variables
Сtrainable_variables
бlayer_metrics
 вlayer_regularization_losses
гlayers
дmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_5/kernel
:2dense_5/bias
0
У0
Ф1"
trackable_list_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
И
Х	variables
Цregularization_losses
еnon_trainable_variables
Чtrainable_variables
жlayer_metrics
 зlayer_regularization_losses
иlayers
йmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
z
20
31
A2
B3
^4
_5
m6
n7
8
9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ў
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ѕ0
І1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ї0
Ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Љ0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
к2з
C__inference_model_1_layer_call_and_return_conditional_losses_238280
C__inference_model_1_layer_call_and_return_conditional_losses_236606
C__inference_model_1_layer_call_and_return_conditional_losses_238652
C__inference_model_1_layer_call_and_return_conditional_losses_236286Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
(__inference_model_1_layer_call_fn_237449
(__inference_model_1_layer_call_fn_238854
(__inference_model_1_layer_call_fn_238753
(__inference_model_1_layer_call_fn_237028Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ч2ф
!__inference__wrapped_model_234145О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_2џџџџџџџџџ22
Ѓ2 
D__inference_conv2d_9_layer_call_and_return_conditional_losses_234172з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_conv2d_9_layer_call_fn_234182з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Є2Ё
E__inference_conv2d_10_layer_call_and_return_conditional_losses_234210з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
*__inference_conv2d_10_layer_call_fn_234220з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239004
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239022
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238929
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238947Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
6__inference_batch_normalization_6_layer_call_fn_239048
6__inference_batch_normalization_6_layer_call_fn_239035
6__inference_batch_normalization_6_layer_call_fn_238973
6__inference_batch_normalization_6_layer_call_fn_238960Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
E__inference_conv2d_11_layer_call_and_return_conditional_losses_234373з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
*__inference_conv2d_11_layer_call_fn_234383з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239107
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239182
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239200
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239125Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
6__inference_batch_normalization_7_layer_call_fn_239138
6__inference_batch_normalization_7_layer_call_fn_239213
6__inference_batch_normalization_7_layer_call_fn_239151
6__inference_batch_normalization_7_layer_call_fn_239226Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ы2ш
A__inference_add_3_layer_call_and_return_conditional_losses_239232Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_add_3_layer_call_fn_239238Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_3_layer_call_and_return_conditional_losses_239243Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_3_layer_call_fn_239248Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
E__inference_conv2d_12_layer_call_and_return_conditional_losses_234537з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
*__inference_conv2d_12_layer_call_fn_234547з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Є2Ё
E__inference_conv2d_13_layer_call_and_return_conditional_losses_234575з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
*__inference_conv2d_13_layer_call_fn_234585з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239341
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239398
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239416
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239323Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
6__inference_batch_normalization_8_layer_call_fn_239442
6__inference_batch_normalization_8_layer_call_fn_239429
6__inference_batch_normalization_8_layer_call_fn_239354
6__inference_batch_normalization_8_layer_call_fn_239367Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
E__inference_conv2d_14_layer_call_and_return_conditional_losses_234738з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
*__inference_conv2d_14_layer_call_fn_234748з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239501
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239576
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239519
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239594Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
6__inference_batch_normalization_9_layer_call_fn_239620
6__inference_batch_normalization_9_layer_call_fn_239545
6__inference_batch_normalization_9_layer_call_fn_239607
6__inference_batch_normalization_9_layer_call_fn_239532Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ы2ш
A__inference_add_4_layer_call_and_return_conditional_losses_239626Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_add_4_layer_call_fn_239632Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_4_layer_call_and_return_conditional_losses_239637Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_4_layer_call_fn_239642Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
E__inference_conv2d_15_layer_call_and_return_conditional_losses_234902з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
*__inference_conv2d_15_layer_call_fn_234912з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Є2Ё
E__inference_conv2d_16_layer_call_and_return_conditional_losses_234940з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
*__inference_conv2d_16_layer_call_fn_234950з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239735
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239792
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239717
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239810Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
7__inference_batch_normalization_10_layer_call_fn_239823
7__inference_batch_normalization_10_layer_call_fn_239761
7__inference_batch_normalization_10_layer_call_fn_239836
7__inference_batch_normalization_10_layer_call_fn_239748Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
E__inference_conv2d_17_layer_call_and_return_conditional_losses_235103з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
*__inference_conv2d_17_layer_call_fn_235113з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239988
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239895
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239970
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239913Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
7__inference_batch_normalization_11_layer_call_fn_239939
7__inference_batch_normalization_11_layer_call_fn_239926
7__inference_batch_normalization_11_layer_call_fn_240014
7__inference_batch_normalization_11_layer_call_fn_240001Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ы2ш
A__inference_add_5_layer_call_and_return_conditional_losses_240020Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_add_5_layer_call_fn_240026Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_5_layer_call_and_return_conditional_losses_240031Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_5_layer_call_fn_240036Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О2Л
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_235246р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ѓ2 
;__inference_global_average_pooling2d_1_layer_call_fn_235252р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
я2ь
E__inference_flatten_1_layer_call_and_return_conditional_losses_240042Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_1_layer_call_fn_240047Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_3_layer_call_and_return_conditional_losses_240090Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_3_layer_call_fn_240099Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_2_layer_call_and_return_conditional_losses_240116
E__inference_dropout_2_layer_call_and_return_conditional_losses_240111Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_2_layer_call_fn_240121
*__inference_dropout_2_layer_call_fn_240126Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
C__inference_dense_4_layer_call_and_return_conditional_losses_240169Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_4_layer_call_fn_240178Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_3_layer_call_and_return_conditional_losses_240195
E__inference_dropout_3_layer_call_and_return_conditional_losses_240190Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_3_layer_call_fn_240200
*__inference_dropout_3_layer_call_fn_240205Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
C__inference_dense_5_layer_call_and_return_conditional_losses_240248Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_5_layer_call_fn_240257Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Г2А
__inference_loss_fn_0_240270
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_1_240283
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_2_240296
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_3_240309
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_4_240322
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_5_240335
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_6_240348
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_7_240361
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_8_240374
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_9_240387
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_10_240400
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_11_240413
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_12_240426
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_13_240439
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_14_240452
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_15_240465
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_16_240478
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_17_240491
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_18_240504
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_19_240517
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_20_240530
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_21_240543
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_22_240556
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_23_240569
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
3B1
$__inference_signature_wrapper_237816input_2з
!__inference__wrapped_model_234145БB#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ8Ђ5
.Ђ+
)&
input_2џџџџџџџџџ22
Њ "1Њ.
,
dense_5!
dense_5џџџџџџџџџД
H__inference_activation_3_layer_call_and_return_conditional_losses_239243h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ "-Ђ*
# 
0џџџџџџџџџ22
 
-__inference_activation_3_layer_call_fn_239248[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22
Њ " џџџџџџџџџ22Д
H__inference_activation_4_layer_call_and_return_conditional_losses_239637h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 
-__inference_activation_4_layer_call_fn_239642[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22 
Њ " џџџџџџџџџ22 Д
H__inference_activation_5_layer_call_and_return_conditional_losses_240031h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22@
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 
-__inference_activation_5_layer_call_fn_240036[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ22@
Њ " џџџџџџџџџ22@с
A__inference_add_3_layer_call_and_return_conditional_losses_239232jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22
*'
inputs/1џџџџџџџџџ22
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Й
&__inference_add_3_layer_call_fn_239238jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22
*'
inputs/1џџџџџџџџџ22
Њ " џџџџџџџџџ22с
A__inference_add_4_layer_call_and_return_conditional_losses_239626jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22 
*'
inputs/1џџџџџџџџџ22 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Й
&__inference_add_4_layer_call_fn_239632jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22 
*'
inputs/1џџџџџџџџџ22 
Њ " џџџџџџџџџ22 с
A__inference_add_5_layer_call_and_return_conditional_losses_240020jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22@
*'
inputs/1џџџџџџџџџ22@
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Й
&__inference_add_5_layer_call_fn_240026jЂg
`Ђ]
[X
*'
inputs/0џџџџџџџџџ22@
*'
inputs/1џџџџџџџџџ22@
Њ " џџџџџџџџџ22@ё
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239717MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ё
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239735MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ь
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239792v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ь
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_239810v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Щ
7__inference_batch_normalization_10_layer_call_fn_239748MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Щ
7__inference_batch_normalization_10_layer_call_fn_239761MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Є
7__inference_batch_normalization_10_layer_call_fn_239823i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ " џџџџџџџџџ22@Є
7__inference_batch_normalization_10_layer_call_fn_239836i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ " џџџџџџџџџ22@ё
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239895MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ё
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239913MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ь
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239970v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Ь
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_239988v;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22@
 Щ
7__inference_batch_normalization_11_layer_call_fn_239926MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Щ
7__inference_batch_normalization_11_layer_call_fn_239939MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Є
7__inference_batch_normalization_11_layer_call_fn_240001i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p
Њ " џџџџџџџџџ22@Є
7__inference_batch_normalization_11_layer_call_fn_240014i;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22@
p 
Њ " џџџџџџџџџ22@Ч
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238929r0123;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ч
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238947r0123;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22
 ь
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2390040123MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2390220123MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
6__inference_batch_normalization_6_layer_call_fn_238960e0123;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ " џџџџџџџџџ22
6__inference_batch_normalization_6_layer_call_fn_238973e0123;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ " џџџџџџџџџ22Ф
6__inference_batch_normalization_6_layer_call_fn_2390350123MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
6__inference_batch_normalization_6_layer_call_fn_2390480123MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239107?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239125?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239182r?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ч
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_239200r?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22
 Ф
6__inference_batch_normalization_7_layer_call_fn_239138?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
6__inference_batch_normalization_7_layer_call_fn_239151?@ABMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
6__inference_batch_normalization_7_layer_call_fn_239213e?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p
Њ " џџџџџџџџџ22
6__inference_batch_normalization_7_layer_call_fn_239226e?@AB;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22
p 
Њ " џџџџџџџџџ22ь
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239323\]^_MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ь
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239341\]^_MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ч
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239398r\]^_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ч
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_239416r\]^_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ф
6__inference_batch_normalization_8_layer_call_fn_239354\]^_MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ф
6__inference_batch_normalization_8_layer_call_fn_239367\]^_MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
6__inference_batch_normalization_8_layer_call_fn_239429e\]^_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ " џџџџџџџџџ22 
6__inference_batch_normalization_8_layer_call_fn_239442e\]^_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ " џџџџџџџџџ22 ь
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239501klmnMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ь
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239519klmnMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ч
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239576rklmn;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ч
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_239594rklmn;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ22 
 Ф
6__inference_batch_normalization_9_layer_call_fn_239532klmnMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ф
6__inference_batch_normalization_9_layer_call_fn_239545klmnMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
6__inference_batch_normalization_9_layer_call_fn_239607eklmn;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p
Њ " џџџџџџџџџ22 
6__inference_batch_normalization_9_layer_call_fn_239620eklmn;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ22 
p 
Њ " џџџџџџџџџ22 к
E__inference_conv2d_10_layer_call_and_return_conditional_losses_234210)*IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 В
*__inference_conv2d_10_layer_call_fn_234220)*IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџк
E__inference_conv2d_11_layer_call_and_return_conditional_losses_23437389IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 В
*__inference_conv2d_11_layer_call_fn_23438389IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџк
E__inference_conv2d_12_layer_call_and_return_conditional_losses_234537OPIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 В
*__inference_conv2d_12_layer_call_fn_234547OPIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ к
E__inference_conv2d_13_layer_call_and_return_conditional_losses_234575UVIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 В
*__inference_conv2d_13_layer_call_fn_234585UVIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ к
E__inference_conv2d_14_layer_call_and_return_conditional_losses_234738deIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 В
*__inference_conv2d_14_layer_call_fn_234748deIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ к
E__inference_conv2d_15_layer_call_and_return_conditional_losses_234902{|IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 В
*__inference_conv2d_15_layer_call_fn_234912{|IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@м
E__inference_conv2d_16_layer_call_and_return_conditional_losses_234940IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Д
*__inference_conv2d_16_layer_call_fn_234950IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@м
E__inference_conv2d_17_layer_call_and_return_conditional_losses_235103IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Д
*__inference_conv2d_17_layer_call_fn_235113IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@й
D__inference_conv2d_9_layer_call_and_return_conditional_losses_234172#$IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_9_layer_call_fn_234182#$IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
C__inference_dense_3_layer_call_and_return_conditional_losses_240090_ЏА/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 ~
(__inference_dense_3_layer_call_fn_240099RЏА/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЇ
C__inference_dense_4_layer_call_and_return_conditional_losses_240169`ЙК0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
(__inference_dense_4_layer_call_fn_240178SЙК0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
C__inference_dense_5_layer_call_and_return_conditional_losses_240248_УФ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
(__inference_dense_5_layer_call_fn_240257RУФ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dropout_2_layer_call_and_return_conditional_losses_240111^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_dropout_2_layer_call_and_return_conditional_losses_240116^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dropout_2_layer_call_fn_240121Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
*__inference_dropout_2_layer_call_fn_240126Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЇ
E__inference_dropout_3_layer_call_and_return_conditional_losses_240190^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_dropout_3_layer_call_and_return_conditional_losses_240195^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dropout_3_layer_call_fn_240200Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
*__inference_dropout_3_layer_call_fn_240205Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЁ
E__inference_flatten_1_layer_call_and_return_conditional_losses_240042X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 y
*__inference_flatten_1_layer_call_fn_240047K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@п
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_235246RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Ж
;__inference_global_average_pooling2d_1_layer_call_fn_235252wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ;
__inference_loss_fn_0_240270#Ђ

Ђ 
Њ " <
__inference_loss_fn_10_240400dЂ

Ђ 
Њ " <
__inference_loss_fn_11_240413eЂ

Ђ 
Њ " <
__inference_loss_fn_12_240426{Ђ

Ђ 
Њ " <
__inference_loss_fn_13_240439|Ђ

Ђ 
Њ " =
__inference_loss_fn_14_240452Ђ

Ђ 
Њ " =
__inference_loss_fn_15_240465Ђ

Ђ 
Њ " =
__inference_loss_fn_16_240478Ђ

Ђ 
Њ " =
__inference_loss_fn_17_240491Ђ

Ђ 
Њ " =
__inference_loss_fn_18_240504ЏЂ

Ђ 
Њ " =
__inference_loss_fn_19_240517АЂ

Ђ 
Њ " ;
__inference_loss_fn_1_240283$Ђ

Ђ 
Њ " =
__inference_loss_fn_20_240530ЙЂ

Ђ 
Њ " =
__inference_loss_fn_21_240543КЂ

Ђ 
Њ " =
__inference_loss_fn_22_240556УЂ

Ђ 
Њ " =
__inference_loss_fn_23_240569ФЂ

Ђ 
Њ " ;
__inference_loss_fn_2_240296)Ђ

Ђ 
Њ " ;
__inference_loss_fn_3_240309*Ђ

Ђ 
Њ " ;
__inference_loss_fn_4_2403228Ђ

Ђ 
Њ " ;
__inference_loss_fn_5_2403359Ђ

Ђ 
Њ " ;
__inference_loss_fn_6_240348OЂ

Ђ 
Њ " ;
__inference_loss_fn_7_240361PЂ

Ђ 
Њ " ;
__inference_loss_fn_8_240374UЂ

Ђ 
Њ " ;
__inference_loss_fn_9_240387VЂ

Ђ 
Њ " ѕ
C__inference_model_1_layer_call_and_return_conditional_losses_236286­B#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ѕ
C__inference_model_1_layer_call_and_return_conditional_losses_236606­B#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 є
C__inference_model_1_layer_call_and_return_conditional_losses_238280ЌB#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p

 
Њ "%Ђ"

0џџџџџџџџџ
 є
C__inference_model_1_layer_call_and_return_conditional_losses_238652ЌB#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Э
(__inference_model_1_layer_call_fn_237028 B#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p

 
Њ "џџџџџџџџџЭ
(__inference_model_1_layer_call_fn_237449 B#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ@Ђ=
6Ђ3
)&
input_2џџџџџџџџџ22
p 

 
Њ "џџџџџџџџџЬ
(__inference_model_1_layer_call_fn_238753B#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p

 
Њ "џџџџџџџџџЬ
(__inference_model_1_layer_call_fn_238854B#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФ?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ22
p 

 
Њ "џџџџџџџџџх
$__inference_signature_wrapper_237816МB#$)*012389?@ABOPUV\]^_deklmn{|ЏАЙКУФCЂ@
Ђ 
9Њ6
4
input_2)&
input_2џџџџџџџџџ22"1Њ.
,
dense_5!
dense_5џџџџџџџџџ