щЧ8
™э
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
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8”®0
Д
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0
Д
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:*
dtype0
Р
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_18/gamma
Й
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_18/beta
З
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_18/moving_mean
Х
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
:*
dtype0
§
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_18/moving_variance
Э
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
:*
dtype0
Д
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:*
dtype0
Р
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_19/gamma
Й
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_19/beta
З
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_19/moving_mean
Х
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
:*
dtype0
§
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_19/moving_variance
Э
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
:*
dtype0
Д
conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_30/kernel
}
$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*&
_output_shapes
: *
dtype0
t
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_30/bias
m
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes
: *
dtype0
Д
conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_31/kernel
}
$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_31/bias
m
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_20/gamma
Й
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_20/beta
З
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_20/moving_mean
Х
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes
: *
dtype0
§
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_20/moving_variance
Э
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes
: *
dtype0
Д
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_21/gamma
Й
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_21/beta
З
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_21/moving_mean
Х
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
: *
dtype0
§
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_21/moving_variance
Э
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
: *
dtype0
Д
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:@*
dtype0
Д
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:@*
dtype0
Р
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_22/gamma
Й
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes
:@*
dtype0
О
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_22/beta
З
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_22/moving_mean
Х
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes
:@*
dtype0
§
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_22/moving_variance
Э
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes
:@*
dtype0
Д
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:@*
dtype0
Р
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_23/gamma
Й
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
:@*
dtype0
О
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_23/beta
З
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_23/moving_mean
Х
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
:@*
dtype0
§
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_23/moving_variance
Э
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
:@*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	@А*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:А*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:А*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	А*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ЇБ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*фА
valueйАBеА BЁА
 
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
Ч
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
Ч
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
Ч
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
Ч
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
А	keras_api
n
Бkernel
	Вbias
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
†
	Зaxis

Иgamma
	Йbeta
Кmoving_mean
Лmoving_variance
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
n
Рkernel
	Сbias
Т	variables
Уregularization_losses
Фtrainable_variables
Х	keras_api
†
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
V
Я	variables
†regularization_losses
°trainable_variables
Ґ	keras_api
V
£	variables
§regularization_losses
•trainable_variables
¶	keras_api
V
І	variables
®regularization_losses
©trainable_variables
™	keras_api
V
Ђ	variables
ђregularization_losses
≠trainable_variables
Ѓ	keras_api
n
ѓkernel
	∞bias
±	variables
≤regularization_losses
≥trainable_variables
і	keras_api
V
µ	variables
ґregularization_losses
Јtrainable_variables
Є	keras_api
n
єkernel
	Їbias
ї	variables
Љregularization_losses
љtrainable_variables
Њ	keras_api
V
њ	variables
јregularization_losses
Ѕtrainable_variables
¬	keras_api
n
√kernel
	ƒbias
≈	variables
∆regularization_losses
«trainable_variables
»	keras_api
И
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
Б30
В31
И32
Й33
К34
Л35
Р36
С37
Ч38
Ш39
Щ40
Ъ41
ѓ42
∞43
є44
Ї45
√46
ƒ47
 
§
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
Б22
В23
И24
Й25
Р26
С27
Ч28
Ш29
ѓ30
∞31
є32
Ї33
√34
ƒ35
≤
	variables
regularization_losses
…non_trainable_variables
 trainable_variables
 layer_metrics
 Ћlayer_regularization_losses
ћlayers
Ќmetrics
 
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
≤
%	variables
&regularization_losses
ќnon_trainable_variables
'trainable_variables
ѕlayer_metrics
 –layer_regularization_losses
—layers
“metrics
\Z
VARIABLE_VALUEconv2d_28/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_28/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
≤
+	variables
,regularization_losses
”non_trainable_variables
-trainable_variables
‘layer_metrics
 ’layer_regularization_losses
÷layers
„metrics
 
ge
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33
 

00
11
≤
4	variables
5regularization_losses
Ўnon_trainable_variables
6trainable_variables
ўlayer_metrics
 Џlayer_regularization_losses
џlayers
№metrics
\Z
VARIABLE_VALUEconv2d_29/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_29/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
≤
:	variables
;regularization_losses
Ёnon_trainable_variables
<trainable_variables
ёlayer_metrics
 яlayer_regularization_losses
аlayers
бmetrics
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2
B3
 

?0
@1
≤
C	variables
Dregularization_losses
вnon_trainable_variables
Etrainable_variables
гlayer_metrics
 дlayer_regularization_losses
еlayers
жmetrics
 
 
 
≤
G	variables
Hregularization_losses
зnon_trainable_variables
Itrainable_variables
иlayer_metrics
 йlayer_regularization_losses
кlayers
лmetrics
 
 
 
≤
K	variables
Lregularization_losses
мnon_trainable_variables
Mtrainable_variables
нlayer_metrics
 оlayer_regularization_losses
пlayers
рmetrics
\Z
VARIABLE_VALUEconv2d_30/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_30/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
≤
Q	variables
Rregularization_losses
сnon_trainable_variables
Strainable_variables
тlayer_metrics
 уlayer_regularization_losses
фlayers
хmetrics
\Z
VARIABLE_VALUEconv2d_31/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_31/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
≤
W	variables
Xregularization_losses
цnon_trainable_variables
Ytrainable_variables
чlayer_metrics
 шlayer_regularization_losses
щlayers
ъmetrics
 
ge
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
^2
_3
 

\0
]1
≤
`	variables
aregularization_losses
ыnon_trainable_variables
btrainable_variables
ьlayer_metrics
 эlayer_regularization_losses
юlayers
€metrics
\Z
VARIABLE_VALUEconv2d_32/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_32/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1
 

d0
e1
≤
f	variables
gregularization_losses
Аnon_trainable_variables
htrainable_variables
Бlayer_metrics
 Вlayer_regularization_losses
Гlayers
Дmetrics
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
m2
n3
 

k0
l1
≤
o	variables
pregularization_losses
Еnon_trainable_variables
qtrainable_variables
Жlayer_metrics
 Зlayer_regularization_losses
Иlayers
Йmetrics
 
 
 
≤
s	variables
tregularization_losses
Кnon_trainable_variables
utrainable_variables
Лlayer_metrics
 Мlayer_regularization_losses
Нlayers
Оmetrics
 
 
 
≤
w	variables
xregularization_losses
Пnon_trainable_variables
ytrainable_variables
Рlayer_metrics
 Сlayer_regularization_losses
Тlayers
Уmetrics
][
VARIABLE_VALUEconv2d_33/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_33/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
 

{0
|1
≤
}	variables
~regularization_losses
Фnon_trainable_variables
trainable_variables
Хlayer_metrics
 Цlayer_regularization_losses
Чlayers
Шmetrics
][
VARIABLE_VALUEconv2d_34/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_34/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Б0
В1
 

Б0
В1
µ
Г	variables
Дregularization_losses
Щnon_trainable_variables
Еtrainable_variables
Ъlayer_metrics
 Ыlayer_regularization_losses
Ьlayers
Эmetrics
 
hf
VARIABLE_VALUEbatch_normalization_22/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_22/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_22/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_22/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
И0
Й1
К2
Л3
 

И0
Й1
µ
М	variables
Нregularization_losses
Юnon_trainable_variables
Оtrainable_variables
Яlayer_metrics
 †layer_regularization_losses
°layers
Ґmetrics
][
VARIABLE_VALUEconv2d_35/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_35/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

Р0
С1
 

Р0
С1
µ
Т	variables
Уregularization_losses
£non_trainable_variables
Фtrainable_variables
§layer_metrics
 •layer_regularization_losses
¶layers
Іmetrics
 
hf
VARIABLE_VALUEbatch_normalization_23/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_23/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_23/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_23/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ч0
Ш1
Щ2
Ъ3
 

Ч0
Ш1
µ
Ы	variables
Ьregularization_losses
®non_trainable_variables
Эtrainable_variables
©layer_metrics
 ™layer_regularization_losses
Ђlayers
ђmetrics
 
 
 
µ
Я	variables
†regularization_losses
≠non_trainable_variables
°trainable_variables
Ѓlayer_metrics
 ѓlayer_regularization_losses
∞layers
±metrics
 
 
 
µ
£	variables
§regularization_losses
≤non_trainable_variables
•trainable_variables
≥layer_metrics
 іlayer_regularization_losses
µlayers
ґmetrics
 
 
 
µ
І	variables
®regularization_losses
Јnon_trainable_variables
©trainable_variables
Єlayer_metrics
 єlayer_regularization_losses
Їlayers
їmetrics
 
 
 
µ
Ђ	variables
ђregularization_losses
Љnon_trainable_variables
≠trainable_variables
љlayer_metrics
 Њlayer_regularization_losses
њlayers
јmetrics
[Y
VARIABLE_VALUEdense_9/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_9/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

ѓ0
∞1
 

ѓ0
∞1
µ
±	variables
≤regularization_losses
Ѕnon_trainable_variables
≥trainable_variables
¬layer_metrics
 √layer_regularization_losses
ƒlayers
≈metrics
 
 
 
µ
µ	variables
ґregularization_losses
∆non_trainable_variables
Јtrainable_variables
«layer_metrics
 »layer_regularization_losses
…layers
 metrics
\Z
VARIABLE_VALUEdense_10/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_10/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

є0
Ї1
 

є0
Ї1
µ
ї	variables
Љregularization_losses
Ћnon_trainable_variables
љtrainable_variables
ћlayer_metrics
 Ќlayer_regularization_losses
ќlayers
ѕmetrics
 
 
 
µ
њ	variables
јregularization_losses
–non_trainable_variables
Ѕtrainable_variables
—layer_metrics
 “layer_regularization_losses
”layers
‘metrics
\Z
VARIABLE_VALUEdense_11/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_11/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

√0
ƒ1
 

√0
ƒ1
µ
≈	variables
∆regularization_losses
’non_trainable_variables
«trainable_variables
÷layer_metrics
 „layer_regularization_losses
Ўlayers
ўmetrics
Z
20
31
A2
B3
^4
_5
m6
n7
К8
Л9
Щ10
Ъ11
 
 
ё
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
К0
Л1
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
Щ0
Ъ1
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
К
serving_default_input_4Placeholder*/
_output_shapes
:€€€€€€€€€22*
dtype0*$
shape:€€€€€€€€€22
≠
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_29/kernelconv2d_29/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_32/kernelconv2d_32/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*<
Tin5
321*
Tout
2*'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_478832
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¶
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpConst*=
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
__inference__traced_save_481756
б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_29/kernelconv2d_29/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_32/kernelconv2d_32/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*<
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
"__inference__traced_restore_481912‘У.
љ∆
ъ
C__inference_model_3_layer_call_and_return_conditional_losses_479668

inputs,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource2
.batch_normalization_18_readvariableop_resource4
0batch_normalization_18_readvariableop_1_resourceC
?batch_normalization_18_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource2
.batch_normalization_19_readvariableop_resource4
0batch_normalization_19_readvariableop_1_resourceC
?batch_normalization_19_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_30_conv2d_readvariableop_resource-
)conv2d_30_biasadd_readvariableop_resource,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource2
.batch_normalization_20_readvariableop_resource4
0batch_normalization_20_readvariableop_1_resourceC
?batch_normalization_20_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource2
.batch_normalization_22_readvariableop_resource4
0batch_normalization_22_readvariableop_1_resourceC
?batch_normalization_22_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_35_conv2d_readvariableop_resource-
)conv2d_35_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityИ≥
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOpЅ
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_27/Conv2D™
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp∞
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_27/BiasAdd≥
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_28/Conv2D/ReadVariableOp’
conv2d_28/Conv2DConv2Dconv2d_27/BiasAdd:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_28/Conv2D™
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp∞
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_28/BiasAdd~
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_28/Reluє
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_18/ReadVariableOpњ
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_18/ReadVariableOp_1м
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1к
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3conv2d_28/Relu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3≥
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_29/Conv2D/ReadVariableOpж
conv2d_29/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_29/Conv2D™
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp∞
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_29/BiasAddє
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_19/ReadVariableOpњ
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_19/ReadVariableOp_1м
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1и
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_29/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3Ґ
	add_9/addAddV2+batch_normalization_19/FusedBatchNormV3:y:0conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
	add_9/addw
activation_9/ReluReluadd_9/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€222
activation_9/Relu≥
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_30/Conv2D/ReadVariableOpЏ
conv2d_30/Conv2DConv2Dactivation_9/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_30/Conv2D™
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp∞
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_30/BiasAdd~
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_30/Relu≥
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_31/Conv2D/ReadVariableOp„
conv2d_31/Conv2DConv2Dconv2d_30/Relu:activations:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_31/Conv2D™
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp∞
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_31/BiasAdd~
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_31/Reluє
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_20/ReadVariableOpњ
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_20/ReadVariableOp_1м
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1к
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_31/Relu:activations:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2)
'batch_normalization_20/FusedBatchNormV3≥
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_32/Conv2D/ReadVariableOpж
conv2d_32/Conv2DConv2D+batch_normalization_20/FusedBatchNormV3:y:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_32/Conv2D™
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp∞
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_32/BiasAddє
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_21/ReadVariableOpњ
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_21/ReadVariableOp_1м
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1и
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_32/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2)
'batch_normalization_21/FusedBatchNormV3¶

add_10/addAddV2+batch_normalization_21/FusedBatchNormV3:y:0conv2d_30/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

add_10/addz
activation_10/ReluReluadd_10/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
activation_10/Relu≥
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_33/Conv2D/ReadVariableOpџ
conv2d_33/Conv2DConv2D activation_10/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_33/Conv2D™
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp∞
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_33/BiasAdd~
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_33/Relu≥
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_34/Conv2D/ReadVariableOp„
conv2d_34/Conv2DConv2Dconv2d_33/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_34/Conv2D™
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp∞
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_34/BiasAdd~
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_34/Reluє
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_22/ReadVariableOpњ
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_22/ReadVariableOp_1м
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1к
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_34/Relu:activations:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_22/FusedBatchNormV3≥
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_35/Conv2D/ReadVariableOpж
conv2d_35/Conv2DConv2D+batch_normalization_22/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_35/Conv2D™
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp∞
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_35/BiasAddє
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_23/ReadVariableOpњ
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_23/ReadVariableOp_1м
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1и
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_35/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2)
'batch_normalization_23/FusedBatchNormV3¶

add_11/addAddV2+batch_normalization_23/FusedBatchNormV3:y:0conv2d_33/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

add_11/addz
activation_11/ReluReluadd_11/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
activation_11/ReluЈ
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_3/Mean/reduction_indicesЏ
global_average_pooling2d_3/MeanMean activation_11/Relu:activations:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
global_average_pooling2d_3/Means
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
flatten_3/ConstІ
flatten_3/ReshapeReshape(global_average_pooling2d_3/Mean:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
flatten_3/Reshape¶
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
dense_9/MatMul/ReadVariableOp†
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/MatMul•
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpҐ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/ReluГ
dropout_6/IdentityIdentitydense_9/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_6/Identity™
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOp§
dense_10/MatMulMatMuldropout_6/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/MatMul®
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOp¶
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/ReluД
dropout_7/IdentityIdentitydense_10/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/Identity©
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp£
dense_11/MatMulMatMuldropout_7/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/MatMulІ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp•
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/BiasAdd|
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/Sigmoidў
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/add 
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/addў
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/add 
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addў
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/add 
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/addў
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/add 
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addў
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/add 
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addў
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/add 
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/addў
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/add 
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addў
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/add 
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addў
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/add 
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/addћ
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/add≈
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/add–
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/add»
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addѕ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/add«
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/addh
IdentityIdentitydense_11/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22:::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:€€€€€€€€€22
 
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
…
Л
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480826

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
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
Ђ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_476912

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ю÷
ў
!__inference__wrapped_model_475161
input_44
0model_3_conv2d_27_conv2d_readvariableop_resource5
1model_3_conv2d_27_biasadd_readvariableop_resource4
0model_3_conv2d_28_conv2d_readvariableop_resource5
1model_3_conv2d_28_biasadd_readvariableop_resource:
6model_3_batch_normalization_18_readvariableop_resource<
8model_3_batch_normalization_18_readvariableop_1_resourceK
Gmodel_3_batch_normalization_18_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_29_conv2d_readvariableop_resource5
1model_3_conv2d_29_biasadd_readvariableop_resource:
6model_3_batch_normalization_19_readvariableop_resource<
8model_3_batch_normalization_19_readvariableop_1_resourceK
Gmodel_3_batch_normalization_19_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_30_conv2d_readvariableop_resource5
1model_3_conv2d_30_biasadd_readvariableop_resource4
0model_3_conv2d_31_conv2d_readvariableop_resource5
1model_3_conv2d_31_biasadd_readvariableop_resource:
6model_3_batch_normalization_20_readvariableop_resource<
8model_3_batch_normalization_20_readvariableop_1_resourceK
Gmodel_3_batch_normalization_20_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_32_conv2d_readvariableop_resource5
1model_3_conv2d_32_biasadd_readvariableop_resource:
6model_3_batch_normalization_21_readvariableop_resource<
8model_3_batch_normalization_21_readvariableop_1_resourceK
Gmodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_33_conv2d_readvariableop_resource5
1model_3_conv2d_33_biasadd_readvariableop_resource4
0model_3_conv2d_34_conv2d_readvariableop_resource5
1model_3_conv2d_34_biasadd_readvariableop_resource:
6model_3_batch_normalization_22_readvariableop_resource<
8model_3_batch_normalization_22_readvariableop_1_resourceK
Gmodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource4
0model_3_conv2d_35_conv2d_readvariableop_resource5
1model_3_conv2d_35_biasadd_readvariableop_resource:
6model_3_batch_normalization_23_readvariableop_resource<
8model_3_batch_normalization_23_readvariableop_1_resourceK
Gmodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceM
Imodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource2
.model_3_dense_9_matmul_readvariableop_resource3
/model_3_dense_9_biasadd_readvariableop_resource3
/model_3_dense_10_matmul_readvariableop_resource4
0model_3_dense_10_biasadd_readvariableop_resource3
/model_3_dense_11_matmul_readvariableop_resource4
0model_3_dense_11_biasadd_readvariableop_resource
identityИЋ
'model_3/conv2d_27/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_3/conv2d_27/Conv2D/ReadVariableOpЏ
model_3/conv2d_27/Conv2DConv2Dinput_4/model_3/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
model_3/conv2d_27/Conv2D¬
(model_3/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_27/BiasAdd/ReadVariableOp–
model_3/conv2d_27/BiasAddBiasAdd!model_3/conv2d_27/Conv2D:output:00model_3/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
model_3/conv2d_27/BiasAddЋ
'model_3/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_3/conv2d_28/Conv2D/ReadVariableOpх
model_3/conv2d_28/Conv2DConv2D"model_3/conv2d_27/BiasAdd:output:0/model_3/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
model_3/conv2d_28/Conv2D¬
(model_3/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_28/BiasAdd/ReadVariableOp–
model_3/conv2d_28/BiasAddBiasAdd!model_3/conv2d_28/Conv2D:output:00model_3/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
model_3/conv2d_28/BiasAddЦ
model_3/conv2d_28/ReluRelu"model_3/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
model_3/conv2d_28/Relu—
-model_3/batch_normalization_18/ReadVariableOpReadVariableOp6model_3_batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_3/batch_normalization_18/ReadVariableOp„
/model_3/batch_normalization_18/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_3/batch_normalization_18/ReadVariableOp_1Д
>model_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOpК
@model_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Ґ
/model_3/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_28/Relu:activations:05model_3/batch_normalization_18/ReadVariableOp:value:07model_3/batch_normalization_18/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 21
/model_3/batch_normalization_18/FusedBatchNormV3Ћ
'model_3/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_3/conv2d_29/Conv2D/ReadVariableOpЖ
model_3/conv2d_29/Conv2DConv2D3model_3/batch_normalization_18/FusedBatchNormV3:y:0/model_3/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
model_3/conv2d_29/Conv2D¬
(model_3/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_29/BiasAdd/ReadVariableOp–
model_3/conv2d_29/BiasAddBiasAdd!model_3/conv2d_29/Conv2D:output:00model_3/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
model_3/conv2d_29/BiasAdd—
-model_3/batch_normalization_19/ReadVariableOpReadVariableOp6model_3_batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_3/batch_normalization_19/ReadVariableOp„
/model_3/batch_normalization_19/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype021
/model_3/batch_normalization_19/ReadVariableOp_1Д
>model_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>model_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOpК
@model_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@model_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1†
/model_3/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_29/BiasAdd:output:05model_3/batch_normalization_19/ReadVariableOp:value:07model_3/batch_normalization_19/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 21
/model_3/batch_normalization_19/FusedBatchNormV3¬
model_3/add_9/addAddV23model_3/batch_normalization_19/FusedBatchNormV3:y:0"model_3/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
model_3/add_9/addП
model_3/activation_9/ReluRelumodel_3/add_9/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€222
model_3/activation_9/ReluЋ
'model_3/conv2d_30/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_30/Conv2D/ReadVariableOpъ
model_3/conv2d_30/Conv2DConv2D'model_3/activation_9/Relu:activations:0/model_3/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
model_3/conv2d_30/Conv2D¬
(model_3/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_30/BiasAdd/ReadVariableOp–
model_3/conv2d_30/BiasAddBiasAdd!model_3/conv2d_30/Conv2D:output:00model_3/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/conv2d_30/BiasAddЦ
model_3/conv2d_30/ReluRelu"model_3/conv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/conv2d_30/ReluЋ
'model_3/conv2d_31/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_3/conv2d_31/Conv2D/ReadVariableOpч
model_3/conv2d_31/Conv2DConv2D$model_3/conv2d_30/Relu:activations:0/model_3/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
model_3/conv2d_31/Conv2D¬
(model_3/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_31/BiasAdd/ReadVariableOp–
model_3/conv2d_31/BiasAddBiasAdd!model_3/conv2d_31/Conv2D:output:00model_3/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/conv2d_31/BiasAddЦ
model_3/conv2d_31/ReluRelu"model_3/conv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/conv2d_31/Relu—
-model_3/batch_normalization_20/ReadVariableOpReadVariableOp6model_3_batch_normalization_20_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_3/batch_normalization_20/ReadVariableOp„
/model_3/batch_normalization_20/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_20_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_3/batch_normalization_20/ReadVariableOp_1Д
>model_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOpК
@model_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Ґ
/model_3/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_31/Relu:activations:05model_3/batch_normalization_20/ReadVariableOp:value:07model_3/batch_normalization_20/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 21
/model_3/batch_normalization_20/FusedBatchNormV3Ћ
'model_3/conv2d_32/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_3/conv2d_32/Conv2D/ReadVariableOpЖ
model_3/conv2d_32/Conv2DConv2D3model_3/batch_normalization_20/FusedBatchNormV3:y:0/model_3/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
model_3/conv2d_32/Conv2D¬
(model_3/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_32/BiasAdd/ReadVariableOp–
model_3/conv2d_32/BiasAddBiasAdd!model_3/conv2d_32/Conv2D:output:00model_3/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/conv2d_32/BiasAdd—
-model_3/batch_normalization_21/ReadVariableOpReadVariableOp6model_3_batch_normalization_21_readvariableop_resource*
_output_shapes
: *
dtype02/
-model_3/batch_normalization_21/ReadVariableOp„
/model_3/batch_normalization_21/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_21_readvariableop_1_resource*
_output_shapes
: *
dtype021
/model_3/batch_normalization_21/ReadVariableOp_1Д
>model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpК
@model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1†
/model_3/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_32/BiasAdd:output:05model_3/batch_normalization_21/ReadVariableOp:value:07model_3/batch_normalization_21/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 21
/model_3/batch_normalization_21/FusedBatchNormV3∆
model_3/add_10/addAddV23model_3/batch_normalization_21/FusedBatchNormV3:y:0$model_3/conv2d_30/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/add_10/addТ
model_3/activation_10/ReluRelumodel_3/add_10/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
model_3/activation_10/ReluЋ
'model_3/conv2d_33/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_3/conv2d_33/Conv2D/ReadVariableOpы
model_3/conv2d_33/Conv2DConv2D(model_3/activation_10/Relu:activations:0/model_3/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
model_3/conv2d_33/Conv2D¬
(model_3/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_33/BiasAdd/ReadVariableOp–
model_3/conv2d_33/BiasAddBiasAdd!model_3/conv2d_33/Conv2D:output:00model_3/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/conv2d_33/BiasAddЦ
model_3/conv2d_33/ReluRelu"model_3/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/conv2d_33/ReluЋ
'model_3/conv2d_34/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_3/conv2d_34/Conv2D/ReadVariableOpч
model_3/conv2d_34/Conv2DConv2D$model_3/conv2d_33/Relu:activations:0/model_3/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
model_3/conv2d_34/Conv2D¬
(model_3/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_34/BiasAdd/ReadVariableOp–
model_3/conv2d_34/BiasAddBiasAdd!model_3/conv2d_34/Conv2D:output:00model_3/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/conv2d_34/BiasAddЦ
model_3/conv2d_34/ReluRelu"model_3/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/conv2d_34/Relu—
-model_3/batch_normalization_22/ReadVariableOpReadVariableOp6model_3_batch_normalization_22_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_3/batch_normalization_22/ReadVariableOp„
/model_3/batch_normalization_22/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_22_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_3/batch_normalization_22/ReadVariableOp_1Д
>model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOpК
@model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Ґ
/model_3/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3$model_3/conv2d_34/Relu:activations:05model_3/batch_normalization_22/ReadVariableOp:value:07model_3/batch_normalization_22/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 21
/model_3/batch_normalization_22/FusedBatchNormV3Ћ
'model_3/conv2d_35/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_3/conv2d_35/Conv2D/ReadVariableOpЖ
model_3/conv2d_35/Conv2DConv2D3model_3/batch_normalization_22/FusedBatchNormV3:y:0/model_3/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
model_3/conv2d_35/Conv2D¬
(model_3/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_35/BiasAdd/ReadVariableOp–
model_3/conv2d_35/BiasAddBiasAdd!model_3/conv2d_35/Conv2D:output:00model_3/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/conv2d_35/BiasAdd—
-model_3/batch_normalization_23/ReadVariableOpReadVariableOp6model_3_batch_normalization_23_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_3/batch_normalization_23/ReadVariableOp„
/model_3/batch_normalization_23/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_23_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/model_3/batch_normalization_23/ReadVariableOp_1Д
>model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOpК
@model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1†
/model_3/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_35/BiasAdd:output:05model_3/batch_normalization_23/ReadVariableOp:value:07model_3/batch_normalization_23/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 21
/model_3/batch_normalization_23/FusedBatchNormV3∆
model_3/add_11/addAddV23model_3/batch_normalization_23/FusedBatchNormV3:y:0$model_3/conv2d_33/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/add_11/addТ
model_3/activation_11/ReluRelumodel_3/add_11/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
model_3/activation_11/Relu«
9model_3/global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_3/global_average_pooling2d_3/Mean/reduction_indicesъ
'model_3/global_average_pooling2d_3/MeanMean(model_3/activation_11/Relu:activations:0Bmodel_3/global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'model_3/global_average_pooling2d_3/MeanГ
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
model_3/flatten_3/Const«
model_3/flatten_3/ReshapeReshape0model_3/global_average_pooling2d_3/Mean:output:0 model_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/flatten_3/ReshapeЊ
%model_3/dense_9/MatMul/ReadVariableOpReadVariableOp.model_3_dense_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02'
%model_3/dense_9/MatMul/ReadVariableOpј
model_3/dense_9/MatMulMatMul"model_3/flatten_3/Reshape:output:0-model_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dense_9/MatMulљ
&model_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02(
&model_3/dense_9/BiasAdd/ReadVariableOp¬
model_3/dense_9/BiasAddBiasAdd model_3/dense_9/MatMul:product:0.model_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dense_9/BiasAddЙ
model_3/dense_9/ReluRelu model_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dense_9/ReluЫ
model_3/dropout_6/IdentityIdentity"model_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dropout_6/Identity¬
&model_3/dense_10/MatMul/ReadVariableOpReadVariableOp/model_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02(
&model_3/dense_10/MatMul/ReadVariableOpƒ
model_3/dense_10/MatMulMatMul#model_3/dropout_6/Identity:output:0.model_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dense_10/MatMulј
'model_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'model_3/dense_10/BiasAdd/ReadVariableOp∆
model_3/dense_10/BiasAddBiasAdd!model_3/dense_10/MatMul:product:0/model_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dense_10/BiasAddМ
model_3/dense_10/ReluRelu!model_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dense_10/ReluЬ
model_3/dropout_7/IdentityIdentity#model_3/dense_10/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_3/dropout_7/IdentityЅ
&model_3/dense_11/MatMul/ReadVariableOpReadVariableOp/model_3_dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02(
&model_3/dense_11/MatMul/ReadVariableOp√
model_3/dense_11/MatMulMatMul#model_3/dropout_7/Identity:output:0.model_3/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_3/dense_11/MatMulњ
'model_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/dense_11/BiasAdd/ReadVariableOp≈
model_3/dense_11/BiasAddBiasAdd!model_3/dense_11/MatMul:product:0/model_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_3/dense_11/BiasAddФ
model_3/dense_11/SigmoidSigmoid!model_3/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_3/dense_11/Sigmoidp
IdentityIdentitymodel_3/dense_11/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22:::::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_4:
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
щ
‘
(__inference_model_3_layer_call_fn_479870

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
identityИҐStatefulPartitionedCallѕ
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
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4783662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
и$
ў
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480339

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
ѓ
o
__inference_loss_fn_18_481520=
9dense_9_kernel_regularizer_square_readvariableop_resource
identityИя
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_9_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/adde
IdentityIdentity"dense_9/kernel/Regularizer/add:z:0*
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
Т
Л
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_475716

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
÷
—
$__inference_signature_wrapper_478832
input_4
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
identityИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_4751612
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_4:
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
≤
™
7__inference_batch_normalization_19_layer_call_fn_480242

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4764192
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
Д
n
__inference_loss_fn_23_481585<
8dense_11_bias_regularizer_square_readvariableop_resource
identityИ„
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_11_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/addd
IdentityIdentity!dense_11/bias/Regularizer/add:z:0*
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
и$
ў
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_475320

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
и$
ў
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480123

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
ш
™
7__inference_batch_normalization_18_layer_call_fn_480051

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4753202
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
≤
™
7__inference_batch_normalization_23_layer_call_fn_481030

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4768412
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
…
p
__inference_loss_fn_22_481572>
:dense_11_kernel_regularizer_square_readvariableop_resource
identityИв
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_11_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addf
IdentityIdentity#dense_11/kernel/Regularizer/add:z:0*
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
Т
Л
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_475351

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
ћ
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_481211

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
∞
™
7__inference_batch_normalization_20_layer_call_fn_480445

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4765232
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
…
Л
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_481004

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
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
†$
ў
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_476612

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
г

*__inference_conv2d_28_layer_call_fn_475236

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_4752262
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э
J
.__inference_activation_10_layer_call_fn_480658

inputs
identity∞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_4766862
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
р
’
(__inference_model_3_layer_call_fn_478044
input_4
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
identityИҐStatefulPartitionedCallƒ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4779452
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_4:
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
Т
Л
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480751

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
К
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_481206

inputs
identityИc
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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
†$
ў
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_476823

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
„
e
I__inference_activation_10_layer_call_and_return_conditional_losses_480653

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
Ђ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_481058

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Э
n
__inference_loss_fn_9_481403=
9conv2d_31_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_31_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/adde
IdentityIdentity"conv2d_31/bias/Regularizer/add:z:0*
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
∞
™
7__inference_batch_normalization_22_layer_call_fn_480839

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4767342
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
г

*__inference_conv2d_34_layer_call_fn_475966

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_4759562
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
…
Л
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_476541

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
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
щ
F
*__inference_dropout_7_layer_call_fn_481221

inputs
identity•
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4770532
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
†$
ў
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_476523

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
Ы
I
-__inference_activation_9_layer_call_fn_480264

inputs
identityѓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_4764752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
Ћ
p
__inference_loss_fn_20_481546>
:dense_10_kernel_regularizer_square_readvariableop_resource
identityИг
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_10_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addf
IdentityIdentity#dense_10/kernel/Regularizer/add:z:0*
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
э
~
)__inference_dense_10_layer_call_fn_481194

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_4770202
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ѓ
ђ
D__inference_dense_10_layer_call_and_return_conditional_losses_477020

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relu«
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addњ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_476050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
г

*__inference_conv2d_29_layer_call_fn_475399

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_4753892
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
–
l
B__inference_add_10_layer_call_and_return_conditional_losses_476672

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:€€€€€€€€€22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22 :€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
„
e
I__inference_activation_11_layer_call_and_return_conditional_losses_476897

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
≤
™
7__inference_batch_normalization_21_layer_call_fn_480561

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4766302
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
≥ 
≠
E__inference_conv2d_33_layer_call_and_return_conditional_losses_475918

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Reluѕ
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/addј
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
…
Л
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_476752

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
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
„
m
A__inference_add_9_layer_call_and_return_conditional_losses_480248
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:€€€€€€€€€222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22:€€€€€€€€€22:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/1
„
e
I__inference_activation_11_layer_call_and_return_conditional_losses_481047

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
…
Л
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480432

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
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
г

*__inference_conv2d_35_layer_call_fn_476129

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_4761192
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э
J
.__inference_activation_11_layer_call_fn_481052

inputs
identity∞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_4768972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
п
W
;__inference_global_average_pooling2d_3_layer_call_fn_476268

inputs
identityЊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_4762622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
†$
ў
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_476734

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
±j
М
__inference__traced_save_481756
file_prefix/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6ebf748dacbd4e08bdec2cba43e7c565/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЅ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*”
value…B∆0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesи
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices•
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *>
dtypes4
2202
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ј
_input_shapes•
Ґ: ::::::::::::::: : :  : : : : : :  : : : : : : @:@:@@:@:@:@:@:@:@@:@:@:@:@:@:	@А:А:
АА:А:	А:: 2(
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
:	@А:!,

_output_shapes	
:А:&-"
 
_output_shapes
:
АА:!.

_output_shapes	
:А:%/!

_output_shapes
:	А: 0

_output_shapes
::1

_output_shapes
: 
Ю
o
__inference_loss_fn_17_481507=
9conv2d_35_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_35_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/adde
IdentityIdentity"conv2d_35/bias/Regularizer/add:z:0*
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
„
e
I__inference_activation_10_layer_call_and_return_conditional_losses_476686

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€22 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22 :W S
/
_output_shapes
:€€€€€€€€€22 
 
_user_specified_nameinputs
≤
™
7__inference_batch_normalization_22_layer_call_fn_480852

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4767522
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
∞
™
7__inference_batch_normalization_18_layer_call_fn_479976

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4763122
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
ш
™
7__inference_batch_normalization_19_layer_call_fn_480154

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4754832
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
ъ
Ђ
C__inference_dense_9_layer_call_and_return_conditional_losses_481106

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Reluƒ
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/addљ
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
∞
™
7__inference_batch_normalization_19_layer_call_fn_480229

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4764012
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
ЛУ
Е
C__inference_model_3_layer_call_and_return_conditional_losses_477622
input_4
conv2d_27_477305
conv2d_27_477307
conv2d_28_477310
conv2d_28_477312!
batch_normalization_18_477315!
batch_normalization_18_477317!
batch_normalization_18_477319!
batch_normalization_18_477321
conv2d_29_477324
conv2d_29_477326!
batch_normalization_19_477329!
batch_normalization_19_477331!
batch_normalization_19_477333!
batch_normalization_19_477335
conv2d_30_477340
conv2d_30_477342
conv2d_31_477345
conv2d_31_477347!
batch_normalization_20_477350!
batch_normalization_20_477352!
batch_normalization_20_477354!
batch_normalization_20_477356
conv2d_32_477359
conv2d_32_477361!
batch_normalization_21_477364!
batch_normalization_21_477366!
batch_normalization_21_477368!
batch_normalization_21_477370
conv2d_33_477375
conv2d_33_477377
conv2d_34_477380
conv2d_34_477382!
batch_normalization_22_477385!
batch_normalization_22_477387!
batch_normalization_22_477389!
batch_normalization_22_477391
conv2d_35_477394
conv2d_35_477396!
batch_normalization_23_477399!
batch_normalization_23_477401!
batch_normalization_23_477403!
batch_normalization_23_477405
dense_9_477412
dense_9_477414
dense_10_477418
dense_10_477420
dense_11_477424
dense_11_477426
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐ.batch_normalization_19/StatefulPartitionedCallҐ.batch_normalization_20/StatefulPartitionedCallҐ.batch_normalization_21/StatefulPartitionedCallҐ.batch_normalization_22/StatefulPartitionedCallҐ.batch_normalization_23/StatefulPartitionedCallҐ!conv2d_27/StatefulPartitionedCallҐ!conv2d_28/StatefulPartitionedCallҐ!conv2d_29/StatefulPartitionedCallҐ!conv2d_30/StatefulPartitionedCallҐ!conv2d_31/StatefulPartitionedCallҐ!conv2d_32/StatefulPartitionedCallҐ!conv2d_33/StatefulPartitionedCallҐ!conv2d_34/StatefulPartitionedCallҐ!conv2d_35/StatefulPartitionedCallҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallГ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_27_477305conv2d_27_477307*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_4751882#
!conv2d_27/StatefulPartitionedCall¶
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_477310conv2d_28_477312*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_4752262#
!conv2d_28/StatefulPartitionedCall©
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_18_477315batch_normalization_18_477317batch_normalization_18_477319batch_normalization_18_477321*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_47633020
.batch_normalization_18/StatefulPartitionedCall≥
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_29_477324conv2d_29_477326*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_4753892#
!conv2d_29/StatefulPartitionedCall©
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_19_477329batch_normalization_19_477331batch_normalization_19_477333batch_normalization_19_477335*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_47641920
.batch_normalization_19/StatefulPartitionedCallТ
add_9/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_4764612
add_9/PartitionedCallб
activation_9/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_4764752
activation_9/PartitionedCall°
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_30_477340conv2d_30_477342*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_4755532#
!conv2d_30/StatefulPartitionedCall¶
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_477345conv2d_31_477347*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_4755912#
!conv2d_31/StatefulPartitionedCall©
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_20_477350batch_normalization_20_477352batch_normalization_20_477354batch_normalization_20_477356*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_47654120
.batch_normalization_20/StatefulPartitionedCall≥
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0conv2d_32_477359conv2d_32_477361*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_4757542#
!conv2d_32/StatefulPartitionedCall©
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0batch_normalization_21_477364batch_normalization_21_477366batch_normalization_21_477368batch_normalization_21_477370*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_47663020
.batch_normalization_21/StatefulPartitionedCallХ
add_10/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_4766722
add_10/PartitionedCallе
activation_10/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_4766862
activation_10/PartitionedCallҐ
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0conv2d_33_477375conv2d_33_477377*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_4759182#
!conv2d_33/StatefulPartitionedCall¶
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_477380conv2d_34_477382*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_4759562#
!conv2d_34/StatefulPartitionedCall©
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_22_477385batch_normalization_22_477387batch_normalization_22_477389batch_normalization_22_477391*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_47675220
.batch_normalization_22/StatefulPartitionedCall≥
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_35_477394conv2d_35_477396*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_4761192#
!conv2d_35/StatefulPartitionedCall©
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_23_477399batch_normalization_23_477401batch_normalization_23_477403batch_normalization_23_477405*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47684120
.batch_normalization_23/StatefulPartitionedCallХ
add_11/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_4768832
add_11/PartitionedCallе
activation_11/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_4768972
activation_11/PartitionedCallЛ
*global_average_pooling2d_3/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_4762622,
*global_average_pooling2d_3/PartitionedCallе
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4769122
flatten_3/PartitionedCallН
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_477412dense_9_477414*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4769472!
dense_9/StatefulPartitionedCallџ
dropout_6/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_4769802
dropout_6/PartitionedCallТ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_10_477418dense_10_477420*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_4770202"
 dense_10/StatefulPartitionedCall№
dropout_7/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4770532
dropout_7/PartitionedCallС
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_11_477424dense_11_477426*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_4770932"
 dense_11/StatefulPartitionedCallЅ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_477305*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/add±
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_477307*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/addЅ
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_477310*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/add±
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_477312*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addЅ
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_477324*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/add±
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_477326*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/addЅ
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_477340*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/add±
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_477342*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addЅ
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_477345*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/add±
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_477347*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addЅ
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_477359*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/add±
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_477361*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/addЅ
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_477375*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/add±
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_477377*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addЅ
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_477380*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/add±
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_477382*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addЅ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_477394*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/add±
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_477396*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/addі
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_477412*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/addђ
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_477414*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addЄ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_477418* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addѓ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_477420*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addЈ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_477424*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addЃ
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_477426*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/addѕ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_4:
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
¶
ђ
D__inference_dense_11_layer_call_and_return_conditional_losses_477093

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoid∆
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addЊ
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_475685

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
и$
ў
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_476213

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
ЈЃ
в
C__inference_model_3_layer_call_and_return_conditional_losses_479296

inputs,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource2
.batch_normalization_18_readvariableop_resource4
0batch_normalization_18_readvariableop_1_resourceC
?batch_normalization_18_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource2
.batch_normalization_19_readvariableop_resource4
0batch_normalization_19_readvariableop_1_resourceC
?batch_normalization_19_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_30_conv2d_readvariableop_resource-
)conv2d_30_biasadd_readvariableop_resource,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource2
.batch_normalization_20_readvariableop_resource4
0batch_normalization_20_readvariableop_1_resourceC
?batch_normalization_20_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource2
.batch_normalization_22_readvariableop_resource4
0batch_normalization_22_readvariableop_1_resourceC
?batch_normalization_22_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_35_conv2d_readvariableop_resource-
)conv2d_35_biasadd_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identityИҐ:batch_normalization_18/AssignMovingAvg/AssignSubVariableOpҐ<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOpҐ:batch_normalization_19/AssignMovingAvg/AssignSubVariableOpҐ<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOpҐ:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpҐ<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpҐ:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpҐ<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpҐ:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpҐ<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOpҐ:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpҐ<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp≥
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOpЅ
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_27/Conv2D™
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp∞
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_27/BiasAdd≥
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_28/Conv2D/ReadVariableOp’
conv2d_28/Conv2DConv2Dconv2d_27/BiasAdd:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_28/Conv2D™
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp∞
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_28/BiasAdd~
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_28/Reluє
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_18/ReadVariableOpњ
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_18/ReadVariableOp_1м
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1„
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3conv2d_28/Relu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2)
'batch_normalization_18/FusedBatchNormV3Б
batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_18/Constх
,batch_normalization_18/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_18/AssignMovingAvg/sub/x≤
*batch_normalization_18/AssignMovingAvg/subSub5batch_normalization_18/AssignMovingAvg/sub/x:output:0%batch_normalization_18/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_18/AssignMovingAvg/subк
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_18/AssignMovingAvg/ReadVariableOp—
,batch_normalization_18/AssignMovingAvg/sub_1Sub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_18/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2.
,batch_normalization_18/AssignMovingAvg/sub_1Ї
*batch_normalization_18/AssignMovingAvg/mulMul0batch_normalization_18/AssignMovingAvg/sub_1:z:0.batch_normalization_18/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2,
*batch_normalization_18/AssignMovingAvg/mulи
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp7^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOpы
.batch_normalization_18/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_18/AssignMovingAvg_1/sub/xЇ
,batch_normalization_18/AssignMovingAvg_1/subSub7batch_normalization_18/AssignMovingAvg_1/sub/x:output:0%batch_normalization_18/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_18/AssignMovingAvg_1/subр
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpЁ
.batch_normalization_18/AssignMovingAvg_1/sub_1Sub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_18/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:20
.batch_normalization_18/AssignMovingAvg_1/sub_1ƒ
,batch_normalization_18/AssignMovingAvg_1/mulMul2batch_normalization_18/AssignMovingAvg_1/sub_1:z:00batch_normalization_18/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2.
,batch_normalization_18/AssignMovingAvg_1/mulц
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp≥
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_29/Conv2D/ReadVariableOpж
conv2d_29/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22*
paddingSAME*
strides
2
conv2d_29/Conv2D™
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp∞
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€222
conv2d_29/BiasAddє
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_19/ReadVariableOpњ
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_19/ReadVariableOp_1м
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1’
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_29/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2)
'batch_normalization_19/FusedBatchNormV3Б
batch_normalization_19/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_19/Constх
,batch_normalization_19/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_19/AssignMovingAvg/sub/x≤
*batch_normalization_19/AssignMovingAvg/subSub5batch_normalization_19/AssignMovingAvg/sub/x:output:0%batch_normalization_19/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_19/AssignMovingAvg/subк
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_19/AssignMovingAvg/ReadVariableOp—
,batch_normalization_19/AssignMovingAvg/sub_1Sub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_19/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg/sub_1Ї
*batch_normalization_19/AssignMovingAvg/mulMul0batch_normalization_19/AssignMovingAvg/sub_1:z:0.batch_normalization_19/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/mulи
:batch_normalization_19/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp7^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_19/AssignMovingAvg/AssignSubVariableOpы
.batch_normalization_19/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_19/AssignMovingAvg_1/sub/xЇ
,batch_normalization_19/AssignMovingAvg_1/subSub7batch_normalization_19/AssignMovingAvg_1/sub/x:output:0%batch_normalization_19/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_19/AssignMovingAvg_1/subр
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpЁ
.batch_normalization_19/AssignMovingAvg_1/sub_1Sub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_19/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:20
.batch_normalization_19/AssignMovingAvg_1/sub_1ƒ
,batch_normalization_19/AssignMovingAvg_1/mulMul2batch_normalization_19/AssignMovingAvg_1/sub_1:z:00batch_normalization_19/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/mulц
<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOpҐ
	add_9/addAddV2+batch_normalization_19/FusedBatchNormV3:y:0conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€222
	add_9/addw
activation_9/ReluReluadd_9/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€222
activation_9/Relu≥
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_30/Conv2D/ReadVariableOpЏ
conv2d_30/Conv2DConv2Dactivation_9/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_30/Conv2D™
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp∞
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_30/BiasAdd~
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_30/Relu≥
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_31/Conv2D/ReadVariableOp„
conv2d_31/Conv2DConv2Dconv2d_30/Relu:activations:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_31/Conv2D™
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp∞
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_31/BiasAdd~
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_31/Reluє
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_20/ReadVariableOpњ
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_20/ReadVariableOp_1м
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1„
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_31/Relu:activations:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2)
'batch_normalization_20/FusedBatchNormV3Б
batch_normalization_20/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_20/Constх
,batch_normalization_20/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_20/AssignMovingAvg/sub/x≤
*batch_normalization_20/AssignMovingAvg/subSub5batch_normalization_20/AssignMovingAvg/sub/x:output:0%batch_normalization_20/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_20/AssignMovingAvg/subк
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOp—
,batch_normalization_20/AssignMovingAvg/sub_1Sub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_20/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_20/AssignMovingAvg/sub_1Ї
*batch_normalization_20/AssignMovingAvg/mulMul0batch_normalization_20/AssignMovingAvg/sub_1:z:0.batch_normalization_20/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_20/AssignMovingAvg/mulи
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp7^batch_normalization_20/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpы
.batch_normalization_20/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_20/AssignMovingAvg_1/sub/xЇ
,batch_normalization_20/AssignMovingAvg_1/subSub7batch_normalization_20/AssignMovingAvg_1/sub/x:output:0%batch_normalization_20/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_20/AssignMovingAvg_1/subр
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpЁ
.batch_normalization_20/AssignMovingAvg_1/sub_1Sub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_20/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_20/AssignMovingAvg_1/sub_1ƒ
,batch_normalization_20/AssignMovingAvg_1/mulMul2batch_normalization_20/AssignMovingAvg_1/sub_1:z:00batch_normalization_20/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_20/AssignMovingAvg_1/mulц
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp≥
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_32/Conv2D/ReadVariableOpж
conv2d_32/Conv2DConv2D+batch_normalization_20/FusedBatchNormV3:y:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 *
paddingSAME*
strides
2
conv2d_32/Conv2D™
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp∞
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
conv2d_32/BiasAddє
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_21/ReadVariableOpњ
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_21/ReadVariableOp_1м
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1’
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_32/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2)
'batch_normalization_21/FusedBatchNormV3Б
batch_normalization_21/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_21/Constх
,batch_normalization_21/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_21/AssignMovingAvg/sub/x≤
*batch_normalization_21/AssignMovingAvg/subSub5batch_normalization_21/AssignMovingAvg/sub/x:output:0%batch_normalization_21/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_21/AssignMovingAvg/subк
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_21/AssignMovingAvg/ReadVariableOp—
,batch_normalization_21/AssignMovingAvg/sub_1Sub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_21/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2.
,batch_normalization_21/AssignMovingAvg/sub_1Ї
*batch_normalization_21/AssignMovingAvg/mulMul0batch_normalization_21/AssignMovingAvg/sub_1:z:0.batch_normalization_21/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_21/AssignMovingAvg/mulи
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp7^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpы
.batch_normalization_21/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_21/AssignMovingAvg_1/sub/xЇ
,batch_normalization_21/AssignMovingAvg_1/subSub7batch_normalization_21/AssignMovingAvg_1/sub/x:output:0%batch_normalization_21/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_21/AssignMovingAvg_1/subр
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpЁ
.batch_normalization_21/AssignMovingAvg_1/sub_1Sub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_21/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 20
.batch_normalization_21/AssignMovingAvg_1/sub_1ƒ
,batch_normalization_21/AssignMovingAvg_1/mulMul2batch_normalization_21/AssignMovingAvg_1/sub_1:z:00batch_normalization_21/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_21/AssignMovingAvg_1/mulц
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp¶

add_10/addAddV2+batch_normalization_21/FusedBatchNormV3:y:0conv2d_30/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

add_10/addz
activation_10/ReluReluadd_10/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2
activation_10/Relu≥
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_33/Conv2D/ReadVariableOpџ
conv2d_33/Conv2DConv2D activation_10/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_33/Conv2D™
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp∞
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_33/BiasAdd~
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_33/Relu≥
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_34/Conv2D/ReadVariableOp„
conv2d_34/Conv2DConv2Dconv2d_33/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_34/Conv2D™
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp∞
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_34/BiasAdd~
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_34/Reluє
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_22/ReadVariableOpњ
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_22/ReadVariableOp_1м
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1„
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_34/Relu:activations:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2)
'batch_normalization_22/FusedBatchNormV3Б
batch_normalization_22/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_22/Constх
,batch_normalization_22/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_22/AssignMovingAvg/sub/x≤
*batch_normalization_22/AssignMovingAvg/subSub5batch_normalization_22/AssignMovingAvg/sub/x:output:0%batch_normalization_22/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_22/AssignMovingAvg/subк
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_22/AssignMovingAvg/ReadVariableOp—
,batch_normalization_22/AssignMovingAvg/sub_1Sub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_22/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_22/AssignMovingAvg/sub_1Ї
*batch_normalization_22/AssignMovingAvg/mulMul0batch_normalization_22/AssignMovingAvg/sub_1:z:0.batch_normalization_22/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_22/AssignMovingAvg/mulи
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp7^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpы
.batch_normalization_22/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_22/AssignMovingAvg_1/sub/xЇ
,batch_normalization_22/AssignMovingAvg_1/subSub7batch_normalization_22/AssignMovingAvg_1/sub/x:output:0%batch_normalization_22/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_22/AssignMovingAvg_1/subр
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpЁ
.batch_normalization_22/AssignMovingAvg_1/sub_1Sub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_22/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_22/AssignMovingAvg_1/sub_1ƒ
,batch_normalization_22/AssignMovingAvg_1/mulMul2batch_normalization_22/AssignMovingAvg_1/sub_1:z:00batch_normalization_22/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_22/AssignMovingAvg_1/mulц
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp≥
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_35/Conv2D/ReadVariableOpж
conv2d_35/Conv2DConv2D+batch_normalization_22/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@*
paddingSAME*
strides
2
conv2d_35/Conv2D™
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp∞
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
conv2d_35/BiasAddє
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_23/ReadVariableOpњ
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_23/ReadVariableOp_1м
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1’
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_35/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2)
'batch_normalization_23/FusedBatchNormV3Б
batch_normalization_23/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_23/Constх
,batch_normalization_23/AssignMovingAvg/sub/xConst*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,batch_normalization_23/AssignMovingAvg/sub/x≤
*batch_normalization_23/AssignMovingAvg/subSub5batch_normalization_23/AssignMovingAvg/sub/x:output:0%batch_normalization_23/Const:output:0*
T0*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2,
*batch_normalization_23/AssignMovingAvg/subк
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_23/AssignMovingAvg/ReadVariableOp—
,batch_normalization_23/AssignMovingAvg/sub_1Sub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_23/FusedBatchNormV3:batch_mean:0*
T0*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2.
,batch_normalization_23/AssignMovingAvg/sub_1Ї
*batch_normalization_23/AssignMovingAvg/mulMul0batch_normalization_23/AssignMovingAvg/sub_1:z:0.batch_normalization_23/AssignMovingAvg/sub:z:0*
T0*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2,
*batch_normalization_23/AssignMovingAvg/mulи
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp7^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02<
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpы
.batch_normalization_23/AssignMovingAvg_1/sub/xConst*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?20
.batch_normalization_23/AssignMovingAvg_1/sub/xЇ
,batch_normalization_23/AssignMovingAvg_1/subSub7batch_normalization_23/AssignMovingAvg_1/sub/x:output:0%batch_normalization_23/Const:output:0*
T0*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2.
,batch_normalization_23/AssignMovingAvg_1/subр
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpЁ
.batch_normalization_23/AssignMovingAvg_1/sub_1Sub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_23/FusedBatchNormV3:batch_variance:0*
T0*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@20
.batch_normalization_23/AssignMovingAvg_1/sub_1ƒ
,batch_normalization_23/AssignMovingAvg_1/mulMul2batch_normalization_23/AssignMovingAvg_1/sub_1:z:00batch_normalization_23/AssignMovingAvg_1/sub:z:0*
T0*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2.
,batch_normalization_23/AssignMovingAvg_1/mulц
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource0batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02>
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp¶

add_11/addAddV2+batch_normalization_23/FusedBatchNormV3:y:0conv2d_33/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

add_11/addz
activation_11/ReluReluadd_11/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2
activation_11/ReluЈ
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_3/Mean/reduction_indicesЏ
global_average_pooling2d_3/MeanMean activation_11/Relu:activations:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
global_average_pooling2d_3/Means
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
flatten_3/ConstІ
flatten_3/ReshapeReshape(global_average_pooling2d_3/Mean:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
flatten_3/Reshape¶
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
dense_9/MatMul/ReadVariableOp†
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/MatMul•
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpҐ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/Reluw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_6/dropout/Const¶
dropout_6/dropout/MulMuldense_9/Relu:activations:0 dropout_6/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/Mul|
dropout_6/dropout/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape”
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_6/dropout/random_uniform/RandomUniformЙ
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_6/dropout/GreaterEqual/yз
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_6/dropout/GreaterEqualЮ
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/Cast£
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_6/dropout/Mul_1™
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOp§
dense_10/MatMulMatMuldropout_6/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/MatMul®
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOp¶
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/Reluw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_7/dropout/ConstІ
dropout_7/dropout/MulMuldense_10/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Mul}
dropout_7/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape”
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_7/dropout/random_uniform/RandomUniformЙ
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_7/dropout/GreaterEqual/yз
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_7/dropout/GreaterEqualЮ
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Cast£
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_7/dropout/Mul_1©
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_11/MatMul/ReadVariableOp£
dense_11/MatMulMatMuldropout_7/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/MatMulІ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp•
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/BiasAdd|
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_11/Sigmoidў
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/add 
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/addў
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/add 
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addў
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/add 
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/addў
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/add 
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addў
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/add 
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addў
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/add 
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/addў
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/add 
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addў
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/add 
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addў
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/add 
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/addћ
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/add≈
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/add–
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/add»
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addѕ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/add«
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/add–
IdentityIdentitydense_11/Sigmoid:y:0;^batch_normalization_18/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_19/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_22/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_19/AssignMovingAvg/AssignSubVariableOp:batch_normalization_19/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_19/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
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
ъ
Ђ
C__inference_dense_9_layer_call_and_return_conditional_losses_476947

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Reluƒ
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/addљ
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480733

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
ћ
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_477053

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
и$
ў
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480592

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
ИУ
Д
C__inference_model_3_layer_call_and_return_conditional_losses_478366

inputs
conv2d_27_478049
conv2d_27_478051
conv2d_28_478054
conv2d_28_478056!
batch_normalization_18_478059!
batch_normalization_18_478061!
batch_normalization_18_478063!
batch_normalization_18_478065
conv2d_29_478068
conv2d_29_478070!
batch_normalization_19_478073!
batch_normalization_19_478075!
batch_normalization_19_478077!
batch_normalization_19_478079
conv2d_30_478084
conv2d_30_478086
conv2d_31_478089
conv2d_31_478091!
batch_normalization_20_478094!
batch_normalization_20_478096!
batch_normalization_20_478098!
batch_normalization_20_478100
conv2d_32_478103
conv2d_32_478105!
batch_normalization_21_478108!
batch_normalization_21_478110!
batch_normalization_21_478112!
batch_normalization_21_478114
conv2d_33_478119
conv2d_33_478121
conv2d_34_478124
conv2d_34_478126!
batch_normalization_22_478129!
batch_normalization_22_478131!
batch_normalization_22_478133!
batch_normalization_22_478135
conv2d_35_478138
conv2d_35_478140!
batch_normalization_23_478143!
batch_normalization_23_478145!
batch_normalization_23_478147!
batch_normalization_23_478149
dense_9_478156
dense_9_478158
dense_10_478162
dense_10_478164
dense_11_478168
dense_11_478170
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐ.batch_normalization_19/StatefulPartitionedCallҐ.batch_normalization_20/StatefulPartitionedCallҐ.batch_normalization_21/StatefulPartitionedCallҐ.batch_normalization_22/StatefulPartitionedCallҐ.batch_normalization_23/StatefulPartitionedCallҐ!conv2d_27/StatefulPartitionedCallҐ!conv2d_28/StatefulPartitionedCallҐ!conv2d_29/StatefulPartitionedCallҐ!conv2d_30/StatefulPartitionedCallҐ!conv2d_31/StatefulPartitionedCallҐ!conv2d_32/StatefulPartitionedCallҐ!conv2d_33/StatefulPartitionedCallҐ!conv2d_34/StatefulPartitionedCallҐ!conv2d_35/StatefulPartitionedCallҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallВ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_478049conv2d_27_478051*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_4751882#
!conv2d_27/StatefulPartitionedCall¶
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_478054conv2d_28_478056*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_4752262#
!conv2d_28/StatefulPartitionedCall©
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_18_478059batch_normalization_18_478061batch_normalization_18_478063batch_normalization_18_478065*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_47633020
.batch_normalization_18/StatefulPartitionedCall≥
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_29_478068conv2d_29_478070*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_4753892#
!conv2d_29/StatefulPartitionedCall©
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_19_478073batch_normalization_19_478075batch_normalization_19_478077batch_normalization_19_478079*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_47641920
.batch_normalization_19/StatefulPartitionedCallТ
add_9/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_4764612
add_9/PartitionedCallб
activation_9/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_4764752
activation_9/PartitionedCall°
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_30_478084conv2d_30_478086*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_4755532#
!conv2d_30/StatefulPartitionedCall¶
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_478089conv2d_31_478091*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_4755912#
!conv2d_31/StatefulPartitionedCall©
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_20_478094batch_normalization_20_478096batch_normalization_20_478098batch_normalization_20_478100*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_47654120
.batch_normalization_20/StatefulPartitionedCall≥
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0conv2d_32_478103conv2d_32_478105*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_4757542#
!conv2d_32/StatefulPartitionedCall©
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0batch_normalization_21_478108batch_normalization_21_478110batch_normalization_21_478112batch_normalization_21_478114*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_47663020
.batch_normalization_21/StatefulPartitionedCallХ
add_10/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_4766722
add_10/PartitionedCallе
activation_10/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_4766862
activation_10/PartitionedCallҐ
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0conv2d_33_478119conv2d_33_478121*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_4759182#
!conv2d_33/StatefulPartitionedCall¶
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_478124conv2d_34_478126*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_4759562#
!conv2d_34/StatefulPartitionedCall©
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_22_478129batch_normalization_22_478131batch_normalization_22_478133batch_normalization_22_478135*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_47675220
.batch_normalization_22/StatefulPartitionedCall≥
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_35_478138conv2d_35_478140*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_4761192#
!conv2d_35/StatefulPartitionedCall©
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_23_478143batch_normalization_23_478145batch_normalization_23_478147batch_normalization_23_478149*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47684120
.batch_normalization_23/StatefulPartitionedCallХ
add_11/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_4768832
add_11/PartitionedCallе
activation_11/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_4768972
activation_11/PartitionedCallЛ
*global_average_pooling2d_3/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_4762622,
*global_average_pooling2d_3/PartitionedCallе
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4769122
flatten_3/PartitionedCallН
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_478156dense_9_478158*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4769472!
dense_9/StatefulPartitionedCallџ
dropout_6/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_4769802
dropout_6/PartitionedCallТ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_10_478162dense_10_478164*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_4770202"
 dense_10/StatefulPartitionedCall№
dropout_7/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4770532
dropout_7/PartitionedCallС
 dense_11/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_11_478168dense_11_478170*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_4770932"
 dense_11/StatefulPartitionedCallЅ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_478049*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/add±
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_478051*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/addЅ
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_478054*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/add±
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_478056*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addЅ
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_478068*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/add±
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_478070*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/addЅ
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_478084*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/add±
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_478086*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addЅ
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_478089*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/add±
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_478091*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addЅ
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_478103*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/add±
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_478105*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/addЅ
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_478119*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/add±
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_478121*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addЅ
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_478124*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/add±
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_478126*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addЅ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_478138*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/add±
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_478140*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/addі
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_478156*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/addђ
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_478158*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addЄ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_478162* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addѓ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_478164*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addЈ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_478168*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addЃ
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_478170*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/addѕ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
К
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_477048

inputs
identityИc
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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
†$
ў
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480986

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
¶
ђ
D__inference_dense_11_layer_call_and_return_conditional_losses_481264

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoid∆
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addЊ
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ю
o
__inference_loss_fn_13_481455=
9conv2d_33_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_33_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/adde
IdentityIdentity"conv2d_33/bias/Regularizer/add:z:0*
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
Ж
n
__inference_loss_fn_21_481559<
8dense_10_bias_regularizer_square_readvariableop_resource
identityИЎ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_10_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addd
IdentityIdentity!dense_10/bias/Regularizer/add:z:0*
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
ъ
™
7__inference_batch_normalization_23_layer_call_fn_480955

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4762442
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
ъ
™
7__inference_batch_normalization_18_layer_call_fn_480064

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4753512
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
ы
~
)__inference_dense_11_layer_call_fn_481273

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_4770932
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
Л
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480610

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
Э
n
__inference_loss_fn_3_481325=
9conv2d_28_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_28_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/adde
IdentityIdentity"conv2d_28/bias/Regularizer/add:z:0*
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
г

*__inference_conv2d_30_layer_call_fn_475563

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_4755532
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ъ
™
7__inference_batch_normalization_20_layer_call_fn_480383

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4757162
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
щ
}
(__inference_dense_9_layer_call_fn_481115

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4769472
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ш
™
7__inference_batch_normalization_23_layer_call_fn_480942

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4762132
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
†$
ў
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_476312

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
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
щ
q
__inference_loss_fn_10_481416?
;conv2d_32_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_32_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/addg
IdentityIdentity$conv2d_32/kernel/Regularizer/add:z:0*
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
Т
Л
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480141

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
§
S
'__inference_add_10_layer_call_fn_480648
inputs_0
inputs_1
identityґ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_4766722
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22 :€€€€€€€€€22 :Y U
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/1
х
F
*__inference_flatten_3_layer_call_fn_481063

inputs
identity§
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4769122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
…
Л
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480216

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
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
†$
ў
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480198

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
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
Љ
≠
E__inference_conv2d_35_layer_call_and_return_conditional_losses_476119

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddѕ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/addј
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ю
o
__inference_loss_fn_11_481429=
9conv2d_32_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_32_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/adde
IdentityIdentity"conv2d_32/bias/Regularizer/add:z:0*
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
Љ
≠
E__inference_conv2d_32_layer_call_and_return_conditional_losses_475754

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddѕ
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/addј
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ш
™
7__inference_batch_normalization_22_layer_call_fn_480764

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4760502
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
ъ
™
7__inference_batch_normalization_19_layer_call_fn_480167

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_4755142
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
Ў
n
B__inference_add_10_layer_call_and_return_conditional_losses_480642
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:€€€€€€€€€22 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22 :€€€€€€€€€22 :Y U
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22 
"
_user_specified_name
inputs/1
Е
c
*__inference_dropout_7_layer_call_fn_481216

inputs
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4770482
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
и$
ў
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_475483

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
и$
ў
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_475848

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
Т
Л
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_476081

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
†$
ў
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480414

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
Т
Л
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480929

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
Љ
≠
E__inference_conv2d_29_layer_call_and_return_conditional_losses_475389

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddѕ
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/addј
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
г

*__inference_conv2d_31_layer_call_fn_475601

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_4755912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
н
‘
(__inference_model_3_layer_call_fn_479769

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
identityИҐStatefulPartitionedCall√
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
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"%&'(+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4779452
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
÷
d
H__inference_activation_9_layer_call_and_return_conditional_losses_476475

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
ш
p
__inference_loss_fn_2_481312?
;conv2d_28_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_28_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/addg
IdentityIdentity$conv2d_28/kernel/Regularizer/add:z:0*
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
ћ
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_481132

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѕќ
•
"__inference__traced_restore_481912
file_prefix%
!assignvariableop_conv2d_27_kernel%
!assignvariableop_1_conv2d_27_bias'
#assignvariableop_2_conv2d_28_kernel%
!assignvariableop_3_conv2d_28_bias3
/assignvariableop_4_batch_normalization_18_gamma2
.assignvariableop_5_batch_normalization_18_beta9
5assignvariableop_6_batch_normalization_18_moving_mean=
9assignvariableop_7_batch_normalization_18_moving_variance'
#assignvariableop_8_conv2d_29_kernel%
!assignvariableop_9_conv2d_29_bias4
0assignvariableop_10_batch_normalization_19_gamma3
/assignvariableop_11_batch_normalization_19_beta:
6assignvariableop_12_batch_normalization_19_moving_mean>
:assignvariableop_13_batch_normalization_19_moving_variance(
$assignvariableop_14_conv2d_30_kernel&
"assignvariableop_15_conv2d_30_bias(
$assignvariableop_16_conv2d_31_kernel&
"assignvariableop_17_conv2d_31_bias4
0assignvariableop_18_batch_normalization_20_gamma3
/assignvariableop_19_batch_normalization_20_beta:
6assignvariableop_20_batch_normalization_20_moving_mean>
:assignvariableop_21_batch_normalization_20_moving_variance(
$assignvariableop_22_conv2d_32_kernel&
"assignvariableop_23_conv2d_32_bias4
0assignvariableop_24_batch_normalization_21_gamma3
/assignvariableop_25_batch_normalization_21_beta:
6assignvariableop_26_batch_normalization_21_moving_mean>
:assignvariableop_27_batch_normalization_21_moving_variance(
$assignvariableop_28_conv2d_33_kernel&
"assignvariableop_29_conv2d_33_bias(
$assignvariableop_30_conv2d_34_kernel&
"assignvariableop_31_conv2d_34_bias4
0assignvariableop_32_batch_normalization_22_gamma3
/assignvariableop_33_batch_normalization_22_beta:
6assignvariableop_34_batch_normalization_22_moving_mean>
:assignvariableop_35_batch_normalization_22_moving_variance(
$assignvariableop_36_conv2d_35_kernel&
"assignvariableop_37_conv2d_35_bias4
0assignvariableop_38_batch_normalization_23_gamma3
/assignvariableop_39_batch_normalization_23_beta:
6assignvariableop_40_batch_normalization_23_moving_mean>
:assignvariableop_41_batch_normalization_23_moving_variance&
"assignvariableop_42_dense_9_kernel$
 assignvariableop_43_dense_9_bias'
#assignvariableop_44_dense_10_kernel%
!assignvariableop_45_dense_10_bias'
#assignvariableop_46_dense_11_kernel%
!assignvariableop_47_dense_11_bias
identity_49ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1«
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*”
value…B∆0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesо
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*÷
_output_shapes√
ј::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
2202
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityС
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_27_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ч
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_27_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Щ
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_28_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ч
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_28_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4•
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_18_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_18_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ђ
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_18_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7ѓ
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_18_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Щ
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_29_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ч
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_29_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_19_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11®
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_19_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12ѓ
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_19_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13≥
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_19_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Э
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_30_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ы
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_30_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Э
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_31_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ы
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_31_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18©
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_20_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_20_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20ѓ
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_20_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21≥
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_20_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Э
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_32_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ы
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_32_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24©
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_21_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25®
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_21_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26ѓ
AssignVariableOp_26AssignVariableOp6assignvariableop_26_batch_normalization_21_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27≥
AssignVariableOp_27AssignVariableOp:assignvariableop_27_batch_normalization_21_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Э
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_33_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Ы
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_33_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Э
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_34_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ы
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_34_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32©
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_22_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33®
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_22_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34ѓ
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_22_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35≥
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_22_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Э
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_35_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Ы
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_35_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38©
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_23_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39®
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_23_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40ѓ
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_23_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41≥
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_23_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Ы
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_9_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43Щ
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_9_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44Ь
AssignVariableOp_44AssignVariableOp#assignvariableop_44_dense_10_kernelIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45Ъ
AssignVariableOp_45AssignVariableOp!assignvariableop_45_dense_10_biasIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46Ь
AssignVariableOp_46AssignVariableOp#assignvariableop_46_dense_11_kernelIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47Ъ
AssignVariableOp_47AssignVariableOp!assignvariableop_47_dense_11_biasIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
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
NoOpю
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48Л	
Identity_49IdentityIdentity_48:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_49"#
identity_49Identity_49:output:0*„
_input_shapes≈
¬: ::::::::::::::::::::::::::::::::::::::::::::::::2$
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
≥ 
≠
E__inference_conv2d_34_layer_call_and_return_conditional_losses_475956

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Reluѕ
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/addј
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
÷
d
H__inference_activation_9_layer_call_and_return_conditional_losses_480259

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€222
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
Љ
≠
E__inference_conv2d_27_layer_call_and_return_conditional_losses_475188

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddѕ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/addј
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/add~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480911

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
К
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_481127

inputs
identityИc
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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ГЦ
Ќ
C__inference_model_3_layer_call_and_return_conditional_losses_477302
input_4
conv2d_27_476272
conv2d_27_476274
conv2d_28_476277
conv2d_28_476279!
batch_normalization_18_476357!
batch_normalization_18_476359!
batch_normalization_18_476361!
batch_normalization_18_476363
conv2d_29_476366
conv2d_29_476368!
batch_normalization_19_476446!
batch_normalization_19_476448!
batch_normalization_19_476450!
batch_normalization_19_476452
conv2d_30_476483
conv2d_30_476485
conv2d_31_476488
conv2d_31_476490!
batch_normalization_20_476568!
batch_normalization_20_476570!
batch_normalization_20_476572!
batch_normalization_20_476574
conv2d_32_476577
conv2d_32_476579!
batch_normalization_21_476657!
batch_normalization_21_476659!
batch_normalization_21_476661!
batch_normalization_21_476663
conv2d_33_476694
conv2d_33_476696
conv2d_34_476699
conv2d_34_476701!
batch_normalization_22_476779!
batch_normalization_22_476781!
batch_normalization_22_476783!
batch_normalization_22_476785
conv2d_35_476788
conv2d_35_476790!
batch_normalization_23_476868!
batch_normalization_23_476870!
batch_normalization_23_476872!
batch_normalization_23_476874
dense_9_476958
dense_9_476960
dense_10_477031
dense_10_477033
dense_11_477104
dense_11_477106
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐ.batch_normalization_19/StatefulPartitionedCallҐ.batch_normalization_20/StatefulPartitionedCallҐ.batch_normalization_21/StatefulPartitionedCallҐ.batch_normalization_22/StatefulPartitionedCallҐ.batch_normalization_23/StatefulPartitionedCallҐ!conv2d_27/StatefulPartitionedCallҐ!conv2d_28/StatefulPartitionedCallҐ!conv2d_29/StatefulPartitionedCallҐ!conv2d_30/StatefulPartitionedCallҐ!conv2d_31/StatefulPartitionedCallҐ!conv2d_32/StatefulPartitionedCallҐ!conv2d_33/StatefulPartitionedCallҐ!conv2d_34/StatefulPartitionedCallҐ!conv2d_35/StatefulPartitionedCallҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐ!dropout_6/StatefulPartitionedCallҐ!dropout_7/StatefulPartitionedCallГ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_27_476272conv2d_27_476274*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_4751882#
!conv2d_27/StatefulPartitionedCall¶
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_476277conv2d_28_476279*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_4752262#
!conv2d_28/StatefulPartitionedCallІ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_18_476357batch_normalization_18_476359batch_normalization_18_476361batch_normalization_18_476363*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_47631220
.batch_normalization_18/StatefulPartitionedCall≥
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_29_476366conv2d_29_476368*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_4753892#
!conv2d_29/StatefulPartitionedCallІ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_19_476446batch_normalization_19_476448batch_normalization_19_476450batch_normalization_19_476452*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_47640120
.batch_normalization_19/StatefulPartitionedCallТ
add_9/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_4764612
add_9/PartitionedCallб
activation_9/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_4764752
activation_9/PartitionedCall°
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_30_476483conv2d_30_476485*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_4755532#
!conv2d_30/StatefulPartitionedCall¶
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_476488conv2d_31_476490*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_4755912#
!conv2d_31/StatefulPartitionedCallІ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_20_476568batch_normalization_20_476570batch_normalization_20_476572batch_normalization_20_476574*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_47652320
.batch_normalization_20/StatefulPartitionedCall≥
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0conv2d_32_476577conv2d_32_476579*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_4757542#
!conv2d_32/StatefulPartitionedCallІ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0batch_normalization_21_476657batch_normalization_21_476659batch_normalization_21_476661batch_normalization_21_476663*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_47661220
.batch_normalization_21/StatefulPartitionedCallХ
add_10/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_4766722
add_10/PartitionedCallе
activation_10/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_4766862
activation_10/PartitionedCallҐ
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0conv2d_33_476694conv2d_33_476696*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_4759182#
!conv2d_33/StatefulPartitionedCall¶
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_476699conv2d_34_476701*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_4759562#
!conv2d_34/StatefulPartitionedCallІ
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_22_476779batch_normalization_22_476781batch_normalization_22_476783batch_normalization_22_476785*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_47673420
.batch_normalization_22/StatefulPartitionedCall≥
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_35_476788conv2d_35_476790*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_4761192#
!conv2d_35/StatefulPartitionedCallІ
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_23_476868batch_normalization_23_476870batch_normalization_23_476872batch_normalization_23_476874*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47682320
.batch_normalization_23/StatefulPartitionedCallХ
add_11/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_4768832
add_11/PartitionedCallе
activation_11/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_4768972
activation_11/PartitionedCallЛ
*global_average_pooling2d_3/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_4762622,
*global_average_pooling2d_3/PartitionedCallе
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4769122
flatten_3/PartitionedCallН
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_476958dense_9_476960*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4769472!
dense_9/StatefulPartitionedCallу
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_4769752#
!dropout_6/StatefulPartitionedCallЪ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_10_477031dense_10_477033*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_4770202"
 dense_10/StatefulPartitionedCallШ
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4770482#
!dropout_7/StatefulPartitionedCallЩ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_11_477104dense_11_477106*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_4770932"
 dense_11/StatefulPartitionedCallЅ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_476272*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/add±
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_476274*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/addЅ
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_476277*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/add±
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_476279*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addЅ
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_476366*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/add±
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_476368*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/addЅ
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_476483*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/add±
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_476485*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addЅ
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_476488*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/add±
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_476490*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addЅ
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_476577*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/add±
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_476579*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/addЅ
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_476694*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/add±
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_476696*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addЅ
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_476699*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/add±
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_476701*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addЅ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_476788*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/add±
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_476790*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/addі
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_476958*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/addђ
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_476960*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addЄ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_477031* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addѓ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_477033*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addЈ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_477104*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addЃ
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_477106*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/addЧ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_4:
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
ъ
™
7__inference_batch_normalization_21_layer_call_fn_480636

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4758792
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
Ў
n
B__inference_add_11_layer_call_and_return_conditional_losses_481036
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:€€€€€€€€€22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22@:€€€€€€€€€22@:Y U
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/1
ш
™
7__inference_batch_normalization_20_layer_call_fn_480370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4756852
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
Т
Л
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_475879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
Т
Л
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_475514

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
Е
c
*__inference_dropout_6_layer_call_fn_481137

inputs
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_4769752
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Т
Л
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_480038

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
∞
™
7__inference_batch_normalization_23_layer_call_fn_481017

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_4768232
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
ъ
™
7__inference_batch_normalization_22_layer_call_fn_480777

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_4760812
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
≤
™
7__inference_batch_normalization_18_layer_call_fn_479989

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4763302
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
†$
ў
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480808

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22@
 
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
∞
™
7__inference_batch_normalization_21_layer_call_fn_480548

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4766122
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
ш
™
7__inference_batch_normalization_21_layer_call_fn_480623

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_4758482
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
Ю
o
__inference_loss_fn_15_481481=
9conv2d_34_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_34_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/adde
IdentityIdentity"conv2d_34/bias/Regularizer/add:z:0*
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
ш
p
__inference_loss_fn_8_481390?
;conv2d_31_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_31_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/addg
IdentityIdentity$conv2d_31/kernel/Regularizer/add:z:0*
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
ш
p
__inference_loss_fn_6_481364?
;conv2d_30_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_30_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/addg
IdentityIdentity$conv2d_30/kernel/Regularizer/add:z:0*
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
…
Л
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_476330

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
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
Ґ
R
&__inference_add_9_layer_call_fn_480254
inputs_0
inputs_1
identityµ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_4764612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22:€€€€€€€€€22:Y U
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22
"
_user_specified_name
inputs/1
≥ 
≠
E__inference_conv2d_30_layer_call_and_return_conditional_losses_475553

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
Reluѕ
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/addј
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
†$
ў
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_479945

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
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
–
l
B__inference_add_11_layer_call_and_return_conditional_losses_476883

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:€€€€€€€€€22@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22@:€€€€€€€€€22@:W S
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€22@
 
_user_specified_nameinputs
г

*__inference_conv2d_27_layer_call_fn_475198

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_4751882
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э
n
__inference_loss_fn_7_481377=
9conv2d_30_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_30_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/adde
IdentityIdentity"conv2d_30/bias/Regularizer/add:z:0*
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
…
Л
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_476630

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
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
К
d
E__inference_dropout_6_layer_call_and_return_conditional_losses_476975

inputs
identityИc
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
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
г

*__inference_conv2d_32_layer_call_fn_475764

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_4757542
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
Л
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_476244

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
§
S
'__inference_add_11_layer_call_fn_481042
inputs_0
inputs_1
identityґ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_4768832
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22@:€€€€€€€€€22@:Y U
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€22@
"
_user_specified_name
inputs/1
щ
q
__inference_loss_fn_16_481494?
;conv2d_35_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_35_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/addg
IdentityIdentity$conv2d_35/kernel/Regularizer/add:z:0*
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
…
Л
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_479963

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
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
Ѓ
ђ
D__inference_dense_10_layer_call_and_return_conditional_losses_481185

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relu«
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addњ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addg
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
м
m
__inference_loss_fn_19_481533;
7dense_9_bias_regularizer_square_readvariableop_resource
identityИ’
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_9_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addc
IdentityIdentity dense_9/bias/Regularizer/add:z:0*
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
†$
ў
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480517

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
…
Л
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_476841

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22@:::::W S
/
_output_shapes
:€€€€€€€€€22@
 
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
щ
q
__inference_loss_fn_12_481442?
;conv2d_33_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_33_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/addg
IdentityIdentity$conv2d_33/kernel/Regularizer/add:z:0*
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
Э
n
__inference_loss_fn_1_481299=
9conv2d_27_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_27_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/adde
IdentityIdentity"conv2d_27/bias/Regularizer/add:z:0*
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
Љ
r
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_476262

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Э
n
__inference_loss_fn_5_481351=
9conv2d_29_bias_regularizer_square_readvariableop_resource
identityИЏ
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_29_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/adde
IdentityIdentity"conv2d_29/bias/Regularizer/add:z:0*
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
…
Л
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_476419

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22:::::W S
/
_output_shapes
:€€€€€€€€€22
 
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
г

*__inference_conv2d_33_layer_call_fn_475928

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_4759182
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ш
p
__inference_loss_fn_4_481338?
;conv2d_29_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_29_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/addg
IdentityIdentity$conv2d_29/kernel/Regularizer/add:z:0*
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
…
Л
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480535

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22 : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 :::::W S
/
_output_shapes
:€€€€€€€€€22 
 
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
Т
Л
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480357

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
щ
F
*__inference_dropout_6_layer_call_fn_481142

inputs
identity•
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_4769802
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ь
’
(__inference_model_3_layer_call_fn_478465
input_4
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
identityИҐStatefulPartitionedCall–
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4783662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€22
!
_user_specified_name	input_4:
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
АЦ
ћ
C__inference_model_3_layer_call_and_return_conditional_losses_477945

inputs
conv2d_27_477628
conv2d_27_477630
conv2d_28_477633
conv2d_28_477635!
batch_normalization_18_477638!
batch_normalization_18_477640!
batch_normalization_18_477642!
batch_normalization_18_477644
conv2d_29_477647
conv2d_29_477649!
batch_normalization_19_477652!
batch_normalization_19_477654!
batch_normalization_19_477656!
batch_normalization_19_477658
conv2d_30_477663
conv2d_30_477665
conv2d_31_477668
conv2d_31_477670!
batch_normalization_20_477673!
batch_normalization_20_477675!
batch_normalization_20_477677!
batch_normalization_20_477679
conv2d_32_477682
conv2d_32_477684!
batch_normalization_21_477687!
batch_normalization_21_477689!
batch_normalization_21_477691!
batch_normalization_21_477693
conv2d_33_477698
conv2d_33_477700
conv2d_34_477703
conv2d_34_477705!
batch_normalization_22_477708!
batch_normalization_22_477710!
batch_normalization_22_477712!
batch_normalization_22_477714
conv2d_35_477717
conv2d_35_477719!
batch_normalization_23_477722!
batch_normalization_23_477724!
batch_normalization_23_477726!
batch_normalization_23_477728
dense_9_477735
dense_9_477737
dense_10_477741
dense_10_477743
dense_11_477747
dense_11_477749
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐ.batch_normalization_19/StatefulPartitionedCallҐ.batch_normalization_20/StatefulPartitionedCallҐ.batch_normalization_21/StatefulPartitionedCallҐ.batch_normalization_22/StatefulPartitionedCallҐ.batch_normalization_23/StatefulPartitionedCallҐ!conv2d_27/StatefulPartitionedCallҐ!conv2d_28/StatefulPartitionedCallҐ!conv2d_29/StatefulPartitionedCallҐ!conv2d_30/StatefulPartitionedCallҐ!conv2d_31/StatefulPartitionedCallҐ!conv2d_32/StatefulPartitionedCallҐ!conv2d_33/StatefulPartitionedCallҐ!conv2d_34/StatefulPartitionedCallҐ!conv2d_35/StatefulPartitionedCallҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐ!dropout_6/StatefulPartitionedCallҐ!dropout_7/StatefulPartitionedCallВ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_477628conv2d_27_477630*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_4751882#
!conv2d_27/StatefulPartitionedCall¶
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_477633conv2d_28_477635*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_4752262#
!conv2d_28/StatefulPartitionedCallІ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_18_477638batch_normalization_18_477640batch_normalization_18_477642batch_normalization_18_477644*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_47631220
.batch_normalization_18/StatefulPartitionedCall≥
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_29_477647conv2d_29_477649*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_4753892#
!conv2d_29/StatefulPartitionedCallІ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_19_477652batch_normalization_19_477654batch_normalization_19_477656batch_normalization_19_477658*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_47640120
.batch_normalization_19/StatefulPartitionedCallТ
add_9/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_4764612
add_9/PartitionedCallб
activation_9/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_4764752
activation_9/PartitionedCall°
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0conv2d_30_477663conv2d_30_477665*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_4755532#
!conv2d_30/StatefulPartitionedCall¶
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_477668conv2d_31_477670*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_4755912#
!conv2d_31/StatefulPartitionedCallІ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_20_477673batch_normalization_20_477675batch_normalization_20_477677batch_normalization_20_477679*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_47652320
.batch_normalization_20/StatefulPartitionedCall≥
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0conv2d_32_477682conv2d_32_477684*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_4757542#
!conv2d_32/StatefulPartitionedCallІ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0batch_normalization_21_477687batch_normalization_21_477689batch_normalization_21_477691batch_normalization_21_477693*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_47661220
.batch_normalization_21/StatefulPartitionedCallХ
add_10/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_4766722
add_10/PartitionedCallе
activation_10/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_10_layer_call_and_return_conditional_losses_4766862
activation_10/PartitionedCallҐ
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_10/PartitionedCall:output:0conv2d_33_477698conv2d_33_477700*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_4759182#
!conv2d_33/StatefulPartitionedCall¶
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_477703conv2d_34_477705*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_4759562#
!conv2d_34/StatefulPartitionedCallІ
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_22_477708batch_normalization_22_477710batch_normalization_22_477712batch_normalization_22_477714*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_47673420
.batch_normalization_22/StatefulPartitionedCall≥
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0conv2d_35_477717conv2d_35_477719*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_4761192#
!conv2d_35/StatefulPartitionedCallІ
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_23_477722batch_normalization_23_477724batch_normalization_23_477726batch_normalization_23_477728*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47682320
.batch_normalization_23/StatefulPartitionedCallХ
add_11/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_4768832
add_11/PartitionedCallе
activation_11/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_activation_11_layer_call_and_return_conditional_losses_4768972
activation_11/PartitionedCallЛ
*global_average_pooling2d_3/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*_
fZRX
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_4762622,
*global_average_pooling2d_3/PartitionedCallе
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4769122
flatten_3/PartitionedCallН
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_477735dense_9_477737*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_4769472!
dense_9/StatefulPartitionedCallу
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_4769752#
!dropout_6/StatefulPartitionedCallЪ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_10_477741dense_10_477743*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_4770202"
 dense_10/StatefulPartitionedCallШ
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4770482#
!dropout_7/StatefulPartitionedCallЩ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_11_477747dense_11_477749*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_4770932"
 dense_11/StatefulPartitionedCallЅ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_477628*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/add±
0conv2d_27/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_477630*
_output_shapes
:*
dtype022
0conv2d_27/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_27/bias/Regularizer/SquareSquare8conv2d_27/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_27/bias/Regularizer/SquareО
 conv2d_27/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_27/bias/Regularizer/ConstЇ
conv2d_27/bias/Regularizer/SumSum%conv2d_27/bias/Regularizer/Square:y:0)conv2d_27/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/SumЙ
 conv2d_27/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_27/bias/Regularizer/mul/xЉ
conv2d_27/bias/Regularizer/mulMul)conv2d_27/bias/Regularizer/mul/x:output:0'conv2d_27/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/mulЙ
 conv2d_27/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_27/bias/Regularizer/add/xє
conv2d_27/bias/Regularizer/addAddV2)conv2d_27/bias/Regularizer/add/x:output:0"conv2d_27/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_27/bias/Regularizer/addЅ
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_477633*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/add±
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_477635*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addЅ
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_477647*&
_output_shapes
:*
dtype024
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_29/kernel/Regularizer/Square°
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_29/kernel/Regularizer/Const¬
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/SumН
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_29/kernel/Regularizer/mul/xƒ
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/mulН
"conv2d_29/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_29/kernel/Regularizer/add/xЅ
 conv2d_29/kernel/Regularizer/addAddV2+conv2d_29/kernel/Regularizer/add/x:output:0$conv2d_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_29/kernel/Regularizer/add±
0conv2d_29/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_477649*
_output_shapes
:*
dtype022
0conv2d_29/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_29/bias/Regularizer/SquareSquare8conv2d_29/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_29/bias/Regularizer/SquareО
 conv2d_29/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_29/bias/Regularizer/ConstЇ
conv2d_29/bias/Regularizer/SumSum%conv2d_29/bias/Regularizer/Square:y:0)conv2d_29/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/SumЙ
 conv2d_29/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_29/bias/Regularizer/mul/xЉ
conv2d_29/bias/Regularizer/mulMul)conv2d_29/bias/Regularizer/mul/x:output:0'conv2d_29/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/mulЙ
 conv2d_29/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_29/bias/Regularizer/add/xє
conv2d_29/bias/Regularizer/addAddV2)conv2d_29/bias/Regularizer/add/x:output:0"conv2d_29/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_29/bias/Regularizer/addЅ
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_477663*&
_output_shapes
: *
dtype024
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_30/kernel/Regularizer/Square°
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_30/kernel/Regularizer/Const¬
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/SumН
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_30/kernel/Regularizer/mul/xƒ
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/mulН
"conv2d_30/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_30/kernel/Regularizer/add/xЅ
 conv2d_30/kernel/Regularizer/addAddV2+conv2d_30/kernel/Regularizer/add/x:output:0$conv2d_30/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_30/kernel/Regularizer/add±
0conv2d_30/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_477665*
_output_shapes
: *
dtype022
0conv2d_30/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_30/bias/Regularizer/SquareSquare8conv2d_30/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_30/bias/Regularizer/SquareО
 conv2d_30/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_30/bias/Regularizer/ConstЇ
conv2d_30/bias/Regularizer/SumSum%conv2d_30/bias/Regularizer/Square:y:0)conv2d_30/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/SumЙ
 conv2d_30/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_30/bias/Regularizer/mul/xЉ
conv2d_30/bias/Regularizer/mulMul)conv2d_30/bias/Regularizer/mul/x:output:0'conv2d_30/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/mulЙ
 conv2d_30/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_30/bias/Regularizer/add/xє
conv2d_30/bias/Regularizer/addAddV2)conv2d_30/bias/Regularizer/add/x:output:0"conv2d_30/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_30/bias/Regularizer/addЅ
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_477668*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/add±
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_477670*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addЅ
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_477682*&
_output_shapes
:  *
dtype024
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_32/kernel/Regularizer/Square°
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_32/kernel/Regularizer/Const¬
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/SumН
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_32/kernel/Regularizer/mul/xƒ
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/mulН
"conv2d_32/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_32/kernel/Regularizer/add/xЅ
 conv2d_32/kernel/Regularizer/addAddV2+conv2d_32/kernel/Regularizer/add/x:output:0$conv2d_32/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_32/kernel/Regularizer/add±
0conv2d_32/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_477684*
_output_shapes
: *
dtype022
0conv2d_32/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_32/bias/Regularizer/SquareSquare8conv2d_32/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_32/bias/Regularizer/SquareО
 conv2d_32/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_32/bias/Regularizer/ConstЇ
conv2d_32/bias/Regularizer/SumSum%conv2d_32/bias/Regularizer/Square:y:0)conv2d_32/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/SumЙ
 conv2d_32/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_32/bias/Regularizer/mul/xЉ
conv2d_32/bias/Regularizer/mulMul)conv2d_32/bias/Regularizer/mul/x:output:0'conv2d_32/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/mulЙ
 conv2d_32/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_32/bias/Regularizer/add/xє
conv2d_32/bias/Regularizer/addAddV2)conv2d_32/bias/Regularizer/add/x:output:0"conv2d_32/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_32/bias/Regularizer/addЅ
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_477698*&
_output_shapes
: @*
dtype024
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_33/kernel/Regularizer/Square°
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_33/kernel/Regularizer/Const¬
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/SumН
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_33/kernel/Regularizer/mul/xƒ
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/mulН
"conv2d_33/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_33/kernel/Regularizer/add/xЅ
 conv2d_33/kernel/Regularizer/addAddV2+conv2d_33/kernel/Regularizer/add/x:output:0$conv2d_33/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_33/kernel/Regularizer/add±
0conv2d_33/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_477700*
_output_shapes
:@*
dtype022
0conv2d_33/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_33/bias/Regularizer/SquareSquare8conv2d_33/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_33/bias/Regularizer/SquareО
 conv2d_33/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_33/bias/Regularizer/ConstЇ
conv2d_33/bias/Regularizer/SumSum%conv2d_33/bias/Regularizer/Square:y:0)conv2d_33/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/SumЙ
 conv2d_33/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_33/bias/Regularizer/mul/xЉ
conv2d_33/bias/Regularizer/mulMul)conv2d_33/bias/Regularizer/mul/x:output:0'conv2d_33/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/mulЙ
 conv2d_33/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_33/bias/Regularizer/add/xє
conv2d_33/bias/Regularizer/addAddV2)conv2d_33/bias/Regularizer/add/x:output:0"conv2d_33/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_33/bias/Regularizer/addЅ
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_477703*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/add±
0conv2d_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_477705*
_output_shapes
:@*
dtype022
0conv2d_34/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_34/bias/Regularizer/SquareSquare8conv2d_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_34/bias/Regularizer/SquareО
 conv2d_34/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_34/bias/Regularizer/ConstЇ
conv2d_34/bias/Regularizer/SumSum%conv2d_34/bias/Regularizer/Square:y:0)conv2d_34/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/SumЙ
 conv2d_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_34/bias/Regularizer/mul/xЉ
conv2d_34/bias/Regularizer/mulMul)conv2d_34/bias/Regularizer/mul/x:output:0'conv2d_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/mulЙ
 conv2d_34/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_34/bias/Regularizer/add/xє
conv2d_34/bias/Regularizer/addAddV2)conv2d_34/bias/Regularizer/add/x:output:0"conv2d_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_34/bias/Regularizer/addЅ
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_477717*&
_output_shapes
:@@*
dtype024
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_35/kernel/Regularizer/Square°
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_35/kernel/Regularizer/Const¬
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/SumН
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_35/kernel/Regularizer/mul/xƒ
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/mulН
"conv2d_35/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_35/kernel/Regularizer/add/xЅ
 conv2d_35/kernel/Regularizer/addAddV2+conv2d_35/kernel/Regularizer/add/x:output:0$conv2d_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_35/kernel/Regularizer/add±
0conv2d_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_477719*
_output_shapes
:@*
dtype022
0conv2d_35/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_35/bias/Regularizer/SquareSquare8conv2d_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!conv2d_35/bias/Regularizer/SquareО
 conv2d_35/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_35/bias/Regularizer/ConstЇ
conv2d_35/bias/Regularizer/SumSum%conv2d_35/bias/Regularizer/Square:y:0)conv2d_35/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/SumЙ
 conv2d_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_35/bias/Regularizer/mul/xЉ
conv2d_35/bias/Regularizer/mulMul)conv2d_35/bias/Regularizer/mul/x:output:0'conv2d_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/mulЙ
 conv2d_35/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_35/bias/Regularizer/add/xє
conv2d_35/bias/Regularizer/addAddV2)conv2d_35/bias/Regularizer/add/x:output:0"conv2d_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_35/bias/Regularizer/addі
0dense_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_477735*
_output_shapes
:	@А*
dtype022
0dense_9/kernel/Regularizer/Square/ReadVariableOpі
!dense_9/kernel/Regularizer/SquareSquare8dense_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@А2#
!dense_9/kernel/Regularizer/SquareХ
 dense_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_9/kernel/Regularizer/ConstЇ
dense_9/kernel/Regularizer/SumSum%dense_9/kernel/Regularizer/Square:y:0)dense_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/SumЙ
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 dense_9/kernel/Regularizer/mul/xЉ
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0'dense_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/mulЙ
 dense_9/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_9/kernel/Regularizer/add/xє
dense_9/kernel/Regularizer/addAddV2)dense_9/kernel/Regularizer/add/x:output:0"dense_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_9/kernel/Regularizer/addђ
.dense_9/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_9_477737*
_output_shapes	
:А*
dtype020
.dense_9/bias/Regularizer/Square/ReadVariableOp™
dense_9/bias/Regularizer/SquareSquare6dense_9/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2!
dense_9/bias/Regularizer/SquareК
dense_9/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_9/bias/Regularizer/Const≤
dense_9/bias/Regularizer/SumSum#dense_9/bias/Regularizer/Square:y:0'dense_9/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/SumЕ
dense_9/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2 
dense_9/bias/Regularizer/mul/xі
dense_9/bias/Regularizer/mulMul'dense_9/bias/Regularizer/mul/x:output:0%dense_9/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/mulЕ
dense_9/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_9/bias/Regularizer/add/x±
dense_9/bias/Regularizer/addAddV2'dense_9/bias/Regularizer/add/x:output:0 dense_9/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_9/bias/Regularizer/addЄ
1dense_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_477741* 
_output_shapes
:
АА*
dtype023
1dense_10/kernel/Regularizer/Square/ReadVariableOpЄ
"dense_10/kernel/Regularizer/SquareSquare9dense_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"dense_10/kernel/Regularizer/SquareЧ
!dense_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_10/kernel/Regularizer/ConstЊ
dense_10/kernel/Regularizer/SumSum&dense_10/kernel/Regularizer/Square:y:0*dense_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/SumЛ
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_10/kernel/Regularizer/mul/xј
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0(dense_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/mulЛ
!dense_10/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_10/kernel/Regularizer/add/xљ
dense_10/kernel/Regularizer/addAddV2*dense_10/kernel/Regularizer/add/x:output:0#dense_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_10/kernel/Regularizer/addѓ
/dense_10/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_10_477743*
_output_shapes	
:А*
dtype021
/dense_10/bias/Regularizer/Square/ReadVariableOp≠
 dense_10/bias/Regularizer/SquareSquare7dense_10/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2"
 dense_10/bias/Regularizer/SquareМ
dense_10/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_10/bias/Regularizer/Constґ
dense_10/bias/Regularizer/SumSum$dense_10/bias/Regularizer/Square:y:0(dense_10/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/SumЗ
dense_10/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_10/bias/Regularizer/mul/xЄ
dense_10/bias/Regularizer/mulMul(dense_10/bias/Regularizer/mul/x:output:0&dense_10/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/mulЗ
dense_10/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_10/bias/Regularizer/add/xµ
dense_10/bias/Regularizer/addAddV2(dense_10/bias/Regularizer/add/x:output:0!dense_10/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_10/bias/Regularizer/addЈ
1dense_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_477747*
_output_shapes
:	А*
dtype023
1dense_11/kernel/Regularizer/Square/ReadVariableOpЈ
"dense_11/kernel/Regularizer/SquareSquare9dense_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2$
"dense_11/kernel/Regularizer/SquareЧ
!dense_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_11/kernel/Regularizer/ConstЊ
dense_11/kernel/Regularizer/SumSum&dense_11/kernel/Regularizer/Square:y:0*dense_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/SumЛ
!dense_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dense_11/kernel/Regularizer/mul/xј
dense_11/kernel/Regularizer/mulMul*dense_11/kernel/Regularizer/mul/x:output:0(dense_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/mulЛ
!dense_11/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_11/kernel/Regularizer/add/xљ
dense_11/kernel/Regularizer/addAddV2*dense_11/kernel/Regularizer/add/x:output:0#dense_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_11/kernel/Regularizer/addЃ
/dense_11/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_11_477749*
_output_shapes
:*
dtype021
/dense_11/bias/Regularizer/Square/ReadVariableOpђ
 dense_11/bias/Regularizer/SquareSquare7dense_11/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_11/bias/Regularizer/SquareМ
dense_11/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_11/bias/Regularizer/Constґ
dense_11/bias/Regularizer/SumSum$dense_11/bias/Regularizer/Square:y:0(dense_11/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/SumЗ
dense_11/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2!
dense_11/bias/Regularizer/mul/xЄ
dense_11/bias/Regularizer/mulMul(dense_11/bias/Regularizer/mul/x:output:0&dense_11/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/mulЗ
dense_11/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_11/bias/Regularizer/add/xµ
dense_11/bias/Regularizer/addAddV2(dense_11/bias/Regularizer/add/x:output:0!dense_11/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_11/bias/Regularizer/addЧ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*р
_input_shapesё
џ:€€€€€€€€€22::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22
 
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
≤
™
7__inference_batch_normalization_20_layer_call_fn_480458

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€22 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_4765412
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€22 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€22 
 
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
ћ
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_476980

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ш
p
__inference_loss_fn_0_481286?
;conv2d_27_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_27_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_27/kernel/Regularizer/Square°
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/Const¬
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/SumН
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_27/kernel/Regularizer/mul/xƒ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulН
"conv2d_27/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_27/kernel/Regularizer/add/xЅ
 conv2d_27/kernel/Regularizer/addAddV2+conv2d_27/kernel/Regularizer/add/x:output:0$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/addg
IdentityIdentity$conv2d_27/kernel/Regularizer/add:z:0*
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
и$
ў
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_480020

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
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
†$
ў
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_476401

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€22:::::*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€22::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€22
 
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
≥ 
≠
E__inference_conv2d_31_layer_call_and_return_conditional_losses_475591

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
Reluѕ
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_31/kernel/Regularizer/Square°
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_31/kernel/Regularizer/Const¬
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/SumН
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_31/kernel/Regularizer/mul/xƒ
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/mulН
"conv2d_31/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_31/kernel/Regularizer/add/xЅ
 conv2d_31/kernel/Regularizer/addAddV2+conv2d_31/kernel/Regularizer/add/x:output:0$conv2d_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_31/kernel/Regularizer/addј
0conv2d_31/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_31/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_31/bias/Regularizer/SquareSquare8conv2d_31/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_31/bias/Regularizer/SquareО
 conv2d_31/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_31/bias/Regularizer/ConstЇ
conv2d_31/bias/Regularizer/SumSum%conv2d_31/bias/Regularizer/Square:y:0)conv2d_31/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/SumЙ
 conv2d_31/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_31/bias/Regularizer/mul/xЉ
conv2d_31/bias/Regularizer/mulMul)conv2d_31/bias/Regularizer/mul/x:output:0'conv2d_31/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/mulЙ
 conv2d_31/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_31/bias/Regularizer/add/xє
conv2d_31/bias/Regularizer/addAddV2)conv2d_31/bias/Regularizer/add/x:output:0"conv2d_31/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_31/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
q
__inference_loss_fn_14_481468?
;conv2d_34_kernel_regularizer_square_readvariableop_resource
identityИм
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_34_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2%
#conv2d_34/kernel/Regularizer/Square°
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_34/kernel/Regularizer/Const¬
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/SumН
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_34/kernel/Regularizer/mul/xƒ
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/mulН
"conv2d_34/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_34/kernel/Regularizer/add/xЅ
 conv2d_34/kernel/Regularizer/addAddV2+conv2d_34/kernel/Regularizer/add/x:output:0$conv2d_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_34/kernel/Regularizer/addg
IdentityIdentity$conv2d_34/kernel/Regularizer/add:z:0*
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
ѕ
k
A__inference_add_9_layer_call_and_return_conditional_losses_476461

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:€€€€€€€€€222
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:€€€€€€€€€222

Identity"
identityIdentity:output:0*I
_input_shapes8
6:€€€€€€€€€22:€€€€€€€€€22:W S
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€22
 
_user_specified_nameinputs
≥ 
≠
E__inference_conv2d_28_layer_call_and_return_conditional_losses_475226

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
Reluѕ
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpЅ
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2%
#conv2d_28/kernel/Regularizer/Square°
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const¬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2$
"conv2d_28/kernel/Regularizer/mul/xƒ
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulН
"conv2d_28/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_28/kernel/Regularizer/add/xЅ
 conv2d_28/kernel/Regularizer/addAddV2+conv2d_28/kernel/Regularizer/add/x:output:0$conv2d_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/addј
0conv2d_28/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv2d_28/bias/Regularizer/Square/ReadVariableOpѓ
!conv2d_28/bias/Regularizer/SquareSquare8conv2d_28/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!conv2d_28/bias/Regularizer/SquareО
 conv2d_28/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_28/bias/Regularizer/ConstЇ
conv2d_28/bias/Regularizer/SumSum%conv2d_28/bias/Regularizer/Square:y:0)conv2d_28/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/SumЙ
 conv2d_28/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2"
 conv2d_28/bias/Regularizer/mul/xЉ
conv2d_28/bias/Regularizer/mulMul)conv2d_28/bias/Regularizer/mul/x:output:0'conv2d_28/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/mulЙ
 conv2d_28/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_28/bias/Regularizer/add/xє
conv2d_28/bias/Regularizer/addAddV2)conv2d_28/bias/Regularizer/add/x:output:0"conv2d_28/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_28/bias/Regularizer/addА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≥
serving_defaultЯ
C
input_48
serving_default_input_4:0€€€€€€€€€22<
dense_110
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Ми
фД
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
+Џ&call_and_return_all_conditional_losses
џ__call__
№_default_save_signature"ћь
_tf_keras_model±ь{"class_name": "Model", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_18", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["batch_normalization_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_9", "trainable": true, "dtype": "float32"}, "name": "add_9", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}], ["conv2d_27", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_9", "inbound_nodes": [[["add_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_30", "inbound_nodes": [[["activation_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_31", "inbound_nodes": [[["conv2d_30", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["conv2d_31", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_32", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_32", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_10", "trainable": true, "dtype": "float32"}, "name": "add_10", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}], ["conv2d_30", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["add_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_33", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["conv2d_33", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["conv2d_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_11", "trainable": true, "dtype": "float32"}, "name": "add_11", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}], ["conv2d_33", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["add_11", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_3", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["global_average_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dropout_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_11", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_18", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["batch_normalization_18", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_9", "trainable": true, "dtype": "float32"}, "name": "add_9", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}], ["conv2d_27", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_9", "inbound_nodes": [[["add_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_30", "inbound_nodes": [[["activation_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_31", "inbound_nodes": [[["conv2d_30", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["conv2d_31", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_32", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_32", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_10", "trainable": true, "dtype": "float32"}, "name": "add_10", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}], ["conv2d_30", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["add_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_33", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["conv2d_33", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["conv2d_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_11", "trainable": true, "dtype": "float32"}, "name": "add_11", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}], ["conv2d_33", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["add_11", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d_3", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["global_average_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_6", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dropout_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_7", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dropout_7", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_11", 0, 0]]}}}
щ"ц
_tf_keras_input_layer÷{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 50, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
–


#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 3]}}
–


)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+я&call_and_return_all_conditional_losses
а__call__"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
Ы	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+б&call_and_return_all_conditional_losses
в__call__"≈
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "batch_normalization_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
“


8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+г&call_and_return_all_conditional_losses
д__call__"Ђ	
_tf_keras_layerС	{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
Ы	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"≈
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
Ш
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+з&call_and_return_all_conditional_losses
и__call__"З
_tf_keras_layerн{"class_name": "Add", "name": "add_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_9", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 16]}, {"class_name": "TensorShape", "items": [null, 50, 50, 16]}]}
і
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+й&call_and_return_all_conditional_losses
к__call__"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "relu"}}
–


Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+л&call_and_return_all_conditional_losses
м__call__"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 16]}}
–


Ukernel
Vbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+н&call_and_return_all_conditional_losses
о__call__"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ы	
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+п&call_and_return_all_conditional_losses
р__call__"≈
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
“


dkernel
ebias
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Ђ	
_tf_keras_layerС	{"class_name": "Conv2D", "name": "conv2d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ы	
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"≈
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
Ъ
s	variables
tregularization_losses
utrainable_variables
v	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"Й
_tf_keras_layerп{"class_name": "Add", "name": "add_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_10", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 32]}, {"class_name": "TensorShape", "items": [null, 50, 50, 32]}]}
ґ
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"•
_tf_keras_layerЛ{"class_name": "Activation", "name": "activation_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}
—


{kernel
|bias
}	variables
~regularization_losses
trainable_variables
А	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 32]}}
÷

Бkernel
	Вbias
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"©	
_tf_keras_layerП	{"class_name": "Conv2D", "name": "conv2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
§	
	Зaxis

Иgamma
	Йbeta
Кmoving_mean
Лmoving_variance
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"≈
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "batch_normalization_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Ў

Рkernel
	Сbias
Т	variables
Уregularization_losses
Фtrainable_variables
Х	keras_api
+€&call_and_return_all_conditional_losses
А__call__"Ђ	
_tf_keras_layerС	{"class_name": "Conv2D", "name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
§	
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"≈
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 50, 64]}}
Ю
Я	variables
†regularization_losses
°trainable_variables
Ґ	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"Й
_tf_keras_layerп{"class_name": "Add", "name": "add_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_11", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 50, 50, 64]}, {"class_name": "TensorShape", "items": [null, 50, 50, 64]}]}
Ї
£	variables
§regularization_losses
•trainable_variables
¶	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"•
_tf_keras_layerЛ{"class_name": "Activation", "name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}
ъ
І	variables
®regularization_losses
©trainable_variables
™	keras_api
+З&call_and_return_all_conditional_losses
И__call__"е
_tf_keras_layerЋ{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling2d_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
…
Ђ	variables
ђregularization_losses
≠trainable_variables
Ѓ	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"і
_tf_keras_layerЪ{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
а
ѓkernel
	∞bias
±	variables
≤regularization_losses
≥trainable_variables
і	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"≥
_tf_keras_layerЩ{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
»
µ	variables
ґregularization_losses
Јtrainable_variables
Є	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
д
єkernel
	Їbias
ї	variables
Љregularization_losses
љtrainable_variables
Њ	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Ј
_tf_keras_layerЭ{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
»
њ	variables
јregularization_losses
Ѕtrainable_variables
¬	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
е
√kernel
	ƒbias
≈	variables
∆regularization_losses
«trainable_variables
»	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Є
_tf_keras_layerЮ{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
®
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
Б30
В31
И32
Й33
К34
Л35
Р36
С37
Ч38
Ш39
Щ40
Ъ41
ѓ42
∞43
є44
Ї45
√46
ƒ47"
trackable_list_wrapper
о
Х0
Ц1
Ч2
Ш3
Щ4
Ъ5
Ы6
Ь7
Э8
Ю9
Я10
†11
°12
Ґ13
£14
§15
•16
¶17
І18
®19
©20
™21
Ђ22
ђ23"
trackable_list_wrapper
ƒ
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
Б22
В23
И24
Й25
Р26
С27
Ч28
Ш29
ѓ30
∞31
є32
Ї33
√34
ƒ35"
trackable_list_wrapper
”
	variables
regularization_losses
…non_trainable_variables
 trainable_variables
 layer_metrics
 Ћlayer_regularization_losses
ћlayers
Ќmetrics
џ__call__
№_default_save_signature
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
-
≠serving_default"
signature_map
*:(2conv2d_27/kernel
:2conv2d_27/bias
.
#0
$1"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
µ
%	variables
&regularization_losses
ќnon_trainable_variables
'trainable_variables
ѕlayer_metrics
 –layer_regularization_losses
—layers
“metrics
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_28/kernel
:2conv2d_28/bias
.
)0
*1"
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
µ
+	variables
,regularization_losses
”non_trainable_variables
-trainable_variables
‘layer_metrics
 ’layer_regularization_losses
÷layers
„metrics
а__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_18/gamma
):'2batch_normalization_18/beta
2:0 (2"batch_normalization_18/moving_mean
6:4 (2&batch_normalization_18/moving_variance
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
µ
4	variables
5regularization_losses
Ўnon_trainable_variables
6trainable_variables
ўlayer_metrics
 Џlayer_regularization_losses
џlayers
№metrics
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_29/kernel
:2conv2d_29/bias
.
80
91"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
:	variables
;regularization_losses
Ёnon_trainable_variables
<trainable_variables
ёlayer_metrics
 яlayer_regularization_losses
аlayers
бmetrics
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_19/gamma
):'2batch_normalization_19/beta
2:0 (2"batch_normalization_19/moving_mean
6:4 (2&batch_normalization_19/moving_variance
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
µ
C	variables
Dregularization_losses
вnon_trainable_variables
Etrainable_variables
гlayer_metrics
 дlayer_regularization_losses
еlayers
жmetrics
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
G	variables
Hregularization_losses
зnon_trainable_variables
Itrainable_variables
иlayer_metrics
 йlayer_regularization_losses
кlayers
лmetrics
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
K	variables
Lregularization_losses
мnon_trainable_variables
Mtrainable_variables
нlayer_metrics
 оlayer_regularization_losses
пlayers
рmetrics
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_30/kernel
: 2conv2d_30/bias
.
O0
P1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
µ
Q	variables
Rregularization_losses
сnon_trainable_variables
Strainable_variables
тlayer_metrics
 уlayer_regularization_losses
фlayers
хmetrics
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_31/kernel
: 2conv2d_31/bias
.
U0
V1"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
µ
W	variables
Xregularization_losses
цnon_trainable_variables
Ytrainable_variables
чlayer_metrics
 шlayer_regularization_losses
щlayers
ъmetrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_20/gamma
):' 2batch_normalization_20/beta
2:0  (2"batch_normalization_20/moving_mean
6:4  (2&batch_normalization_20/moving_variance
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
µ
`	variables
aregularization_losses
ыnon_trainable_variables
btrainable_variables
ьlayer_metrics
 эlayer_regularization_losses
юlayers
€metrics
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_32/kernel
: 2conv2d_32/bias
.
d0
e1"
trackable_list_wrapper
0
Я0
†1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
µ
f	variables
gregularization_losses
Аnon_trainable_variables
htrainable_variables
Бlayer_metrics
 Вlayer_regularization_losses
Гlayers
Дmetrics
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_21/gamma
):' 2batch_normalization_21/beta
2:0  (2"batch_normalization_21/moving_mean
6:4  (2&batch_normalization_21/moving_variance
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
µ
o	variables
pregularization_losses
Еnon_trainable_variables
qtrainable_variables
Жlayer_metrics
 Зlayer_regularization_losses
Иlayers
Йmetrics
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
s	variables
tregularization_losses
Кnon_trainable_variables
utrainable_variables
Лlayer_metrics
 Мlayer_regularization_losses
Нlayers
Оmetrics
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
µ
w	variables
xregularization_losses
Пnon_trainable_variables
ytrainable_variables
Рlayer_metrics
 Сlayer_regularization_losses
Тlayers
Уmetrics
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_33/kernel
:@2conv2d_33/bias
.
{0
|1"
trackable_list_wrapper
0
°0
Ґ1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
µ
}	variables
~regularization_losses
Фnon_trainable_variables
trainable_variables
Хlayer_metrics
 Цlayer_regularization_losses
Чlayers
Шmetrics
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_34/kernel
:@2conv2d_34/bias
0
Б0
В1"
trackable_list_wrapper
0
£0
§1"
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
Є
Г	variables
Дregularization_losses
Щnon_trainable_variables
Еtrainable_variables
Ъlayer_metrics
 Ыlayer_regularization_losses
Ьlayers
Эmetrics
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_22/gamma
):'@2batch_normalization_22/beta
2:0@ (2"batch_normalization_22/moving_mean
6:4@ (2&batch_normalization_22/moving_variance
@
И0
Й1
К2
Л3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
Є
М	variables
Нregularization_losses
Юnon_trainable_variables
Оtrainable_variables
Яlayer_metrics
 †layer_regularization_losses
°layers
Ґmetrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_35/kernel
:@2conv2d_35/bias
0
Р0
С1"
trackable_list_wrapper
0
•0
¶1"
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
Є
Т	variables
Уregularization_losses
£non_trainable_variables
Фtrainable_variables
§layer_metrics
 •layer_regularization_losses
¶layers
Іmetrics
А__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_23/gamma
):'@2batch_normalization_23/beta
2:0@ (2"batch_normalization_23/moving_mean
6:4@ (2&batch_normalization_23/moving_variance
@
Ч0
Ш1
Щ2
Ъ3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
Є
Ы	variables
Ьregularization_losses
®non_trainable_variables
Эtrainable_variables
©layer_metrics
 ™layer_regularization_losses
Ђlayers
ђmetrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Я	variables
†regularization_losses
≠non_trainable_variables
°trainable_variables
Ѓlayer_metrics
 ѓlayer_regularization_losses
∞layers
±metrics
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
£	variables
§regularization_losses
≤non_trainable_variables
•trainable_variables
≥layer_metrics
 іlayer_regularization_losses
µlayers
ґmetrics
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
І	variables
®regularization_losses
Јnon_trainable_variables
©trainable_variables
Єlayer_metrics
 єlayer_regularization_losses
Їlayers
їmetrics
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ђ	variables
ђregularization_losses
Љnon_trainable_variables
≠trainable_variables
љlayer_metrics
 Њlayer_regularization_losses
њlayers
јmetrics
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
!:	@А2dense_9/kernel
:А2dense_9/bias
0
ѓ0
∞1"
trackable_list_wrapper
0
І0
®1"
trackable_list_wrapper
0
ѓ0
∞1"
trackable_list_wrapper
Є
±	variables
≤regularization_losses
Ѕnon_trainable_variables
≥trainable_variables
¬layer_metrics
 √layer_regularization_losses
ƒlayers
≈metrics
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
µ	variables
ґregularization_losses
∆non_trainable_variables
Јtrainable_variables
«layer_metrics
 »layer_regularization_losses
…layers
 metrics
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
#:!
АА2dense_10/kernel
:А2dense_10/bias
0
є0
Ї1"
trackable_list_wrapper
0
©0
™1"
trackable_list_wrapper
0
є0
Ї1"
trackable_list_wrapper
Є
ї	variables
Љregularization_losses
Ћnon_trainable_variables
љtrainable_variables
ћlayer_metrics
 Ќlayer_regularization_losses
ќlayers
ѕmetrics
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
њ	variables
јregularization_losses
–non_trainable_variables
Ѕtrainable_variables
—layer_metrics
 “layer_regularization_losses
”layers
‘metrics
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_11/kernel
:2dense_11/bias
0
√0
ƒ1"
trackable_list_wrapper
0
Ђ0
ђ1"
trackable_list_wrapper
0
√0
ƒ1"
trackable_list_wrapper
Є
≈	variables
∆regularization_losses
’non_trainable_variables
«trainable_variables
÷layer_metrics
 „layer_regularization_losses
Ўlayers
ўmetrics
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
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
К8
Л9
Щ10
Ъ11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ю
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
Х0
Ц1"
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
Ч0
Ш1"
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
Щ0
Ъ1"
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
Ы0
Ь1"
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
Э0
Ю1"
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
Я0
†1"
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
°0
Ґ1"
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
£0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
К0
Л1"
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
•0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Щ0
Ъ1"
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
І0
®1"
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
©0
™1"
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
Ђ0
ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Џ2„
C__inference_model_3_layer_call_and_return_conditional_losses_479668
C__inference_model_3_layer_call_and_return_conditional_losses_479296
C__inference_model_3_layer_call_and_return_conditional_losses_477302
C__inference_model_3_layer_call_and_return_conditional_losses_477622ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
(__inference_model_3_layer_call_fn_479769
(__inference_model_3_layer_call_fn_479870
(__inference_model_3_layer_call_fn_478044
(__inference_model_3_layer_call_fn_478465ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
!__inference__wrapped_model_475161Њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *.Ґ+
)К&
input_4€€€€€€€€€22
§2°
E__inference_conv2d_27_layer_call_and_return_conditional_losses_475188„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Й2Ж
*__inference_conv2d_27_layer_call_fn_475198„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
§2°
E__inference_conv2d_28_layer_call_and_return_conditional_losses_475226„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Й2Ж
*__inference_conv2d_28_layer_call_fn_475236„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
К2З
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_479945
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_479963
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_480038
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_480020і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_18_layer_call_fn_480064
7__inference_batch_normalization_18_layer_call_fn_480051
7__inference_batch_normalization_18_layer_call_fn_479976
7__inference_batch_normalization_18_layer_call_fn_479989і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
E__inference_conv2d_29_layer_call_and_return_conditional_losses_475389„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Й2Ж
*__inference_conv2d_29_layer_call_fn_475399„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
К2З
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480141
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480216
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480198
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480123і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_19_layer_call_fn_480154
7__inference_batch_normalization_19_layer_call_fn_480167
7__inference_batch_normalization_19_layer_call_fn_480229
7__inference_batch_normalization_19_layer_call_fn_480242і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
л2и
A__inference_add_9_layer_call_and_return_conditional_losses_480248Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_add_9_layer_call_fn_480254Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_9_layer_call_and_return_conditional_losses_480259Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_9_layer_call_fn_480264Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
§2°
E__inference_conv2d_30_layer_call_and_return_conditional_losses_475553„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Й2Ж
*__inference_conv2d_30_layer_call_fn_475563„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
§2°
E__inference_conv2d_31_layer_call_and_return_conditional_losses_475591„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Й2Ж
*__inference_conv2d_31_layer_call_fn_475601„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
К2З
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480357
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480414
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480432
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480339і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_20_layer_call_fn_480383
7__inference_batch_normalization_20_layer_call_fn_480445
7__inference_batch_normalization_20_layer_call_fn_480370
7__inference_batch_normalization_20_layer_call_fn_480458і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
E__inference_conv2d_32_layer_call_and_return_conditional_losses_475754„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Й2Ж
*__inference_conv2d_32_layer_call_fn_475764„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
К2З
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480535
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480517
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480610
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480592і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_21_layer_call_fn_480623
7__inference_batch_normalization_21_layer_call_fn_480636
7__inference_batch_normalization_21_layer_call_fn_480561
7__inference_batch_normalization_21_layer_call_fn_480548і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_add_10_layer_call_and_return_conditional_losses_480642Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_add_10_layer_call_fn_480648Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_activation_10_layer_call_and_return_conditional_losses_480653Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_activation_10_layer_call_fn_480658Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
§2°
E__inference_conv2d_33_layer_call_and_return_conditional_losses_475918„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Й2Ж
*__inference_conv2d_33_layer_call_fn_475928„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
§2°
E__inference_conv2d_34_layer_call_and_return_conditional_losses_475956„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Й2Ж
*__inference_conv2d_34_layer_call_fn_475966„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
К2З
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480751
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480826
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480733
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480808і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_22_layer_call_fn_480777
7__inference_batch_normalization_22_layer_call_fn_480839
7__inference_batch_normalization_22_layer_call_fn_480764
7__inference_batch_normalization_22_layer_call_fn_480852і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
§2°
E__inference_conv2d_35_layer_call_and_return_conditional_losses_476119„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Й2Ж
*__inference_conv2d_35_layer_call_fn_476129„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
К2З
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480986
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_481004
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480911
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480929і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
7__inference_batch_normalization_23_layer_call_fn_481017
7__inference_batch_normalization_23_layer_call_fn_481030
7__inference_batch_normalization_23_layer_call_fn_480942
7__inference_batch_normalization_23_layer_call_fn_480955і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_add_11_layer_call_and_return_conditional_losses_481036Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_add_11_layer_call_fn_481042Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_activation_11_layer_call_and_return_conditional_losses_481047Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_activation_11_layer_call_fn_481052Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_476262а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
£2†
;__inference_global_average_pooling2d_3_layer_call_fn_476268а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
п2м
E__inference_flatten_3_layer_call_and_return_conditional_losses_481058Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
*__inference_flatten_3_layer_call_fn_481063Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_9_layer_call_and_return_conditional_losses_481106Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_9_layer_call_fn_481115Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
E__inference_dropout_6_layer_call_and_return_conditional_losses_481132
E__inference_dropout_6_layer_call_and_return_conditional_losses_481127і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_6_layer_call_fn_481137
*__inference_dropout_6_layer_call_fn_481142і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_dense_10_layer_call_and_return_conditional_losses_481185Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_10_layer_call_fn_481194Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»2≈
E__inference_dropout_7_layer_call_and_return_conditional_losses_481206
E__inference_dropout_7_layer_call_and_return_conditional_losses_481211і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
*__inference_dropout_7_layer_call_fn_481216
*__inference_dropout_7_layer_call_fn_481221і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_dense_11_layer_call_and_return_conditional_losses_481264Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_11_layer_call_fn_481273Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≥2∞
__inference_loss_fn_0_481286П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_1_481299П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_2_481312П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_3_481325П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_4_481338П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_5_481351П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_6_481364П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_7_481377П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_8_481390П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_9_481403П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_10_481416П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_11_481429П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_12_481442П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_13_481455П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_14_481468П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_15_481481П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_16_481494П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_17_481507П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_18_481520П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_19_481533П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_20_481546П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_21_481559П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_22_481572П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
і2±
__inference_loss_fn_23_481585П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
3B1
$__inference_signature_wrapper_478832input_4ў
!__inference__wrapped_model_475161≥B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ8Ґ5
.Ґ+
)К&
input_4€€€€€€€€€22
™ "3™0
.
dense_11"К
dense_11€€€€€€€€€µ
I__inference_activation_10_layer_call_and_return_conditional_losses_480653h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ Н
.__inference_activation_10_layer_call_fn_480658[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22 
™ " К€€€€€€€€€22 µ
I__inference_activation_11_layer_call_and_return_conditional_losses_481047h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22@
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ Н
.__inference_activation_11_layer_call_fn_481052[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22@
™ " К€€€€€€€€€22@і
H__inference_activation_9_layer_call_and_return_conditional_losses_480259h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ М
-__inference_activation_9_layer_call_fn_480264[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€22
™ " К€€€€€€€€€22в
B__inference_add_10_layer_call_and_return_conditional_losses_480642ЫjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22 
*К'
inputs/1€€€€€€€€€22 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ Ї
'__inference_add_10_layer_call_fn_480648ОjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22 
*К'
inputs/1€€€€€€€€€22 
™ " К€€€€€€€€€22 в
B__inference_add_11_layer_call_and_return_conditional_losses_481036ЫjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22@
*К'
inputs/1€€€€€€€€€22@
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ Ї
'__inference_add_11_layer_call_fn_481042ОjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22@
*К'
inputs/1€€€€€€€€€22@
™ " К€€€€€€€€€22@б
A__inference_add_9_layer_call_and_return_conditional_losses_480248ЫjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22
*К'
inputs/1€€€€€€€€€22
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ є
&__inference_add_9_layer_call_fn_480254ОjҐg
`Ґ]
[ЪX
*К'
inputs/0€€€€€€€€€22
*К'
inputs/1€€€€€€€€€22
™ " К€€€€€€€€€22»
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_479945r0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ »
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_479963r0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ н
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_480020Ц0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ н
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_480038Ц0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ †
7__inference_batch_normalization_18_layer_call_fn_479976e0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ " К€€€€€€€€€22†
7__inference_batch_normalization_18_layer_call_fn_479989e0123;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ " К€€€€€€€€€22≈
7__inference_batch_normalization_18_layer_call_fn_480051Й0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≈
7__inference_batch_normalization_18_layer_call_fn_480064Й0123MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€н
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480123Ц?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ н
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480141Ц?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480198r?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ »
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_480216r?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ "-Ґ*
#К 
0€€€€€€€€€22
Ъ ≈
7__inference_batch_normalization_19_layer_call_fn_480154Й?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≈
7__inference_batch_normalization_19_layer_call_fn_480167Й?@ABMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€†
7__inference_batch_normalization_19_layer_call_fn_480229e?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p
™ " К€€€€€€€€€22†
7__inference_batch_normalization_19_layer_call_fn_480242e?@AB;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22
p 
™ " К€€€€€€€€€22н
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480339Ц\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ н
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480357Ц\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ »
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480414r\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ »
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_480432r\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ ≈
7__inference_batch_normalization_20_layer_call_fn_480370Й\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ≈
7__inference_batch_normalization_20_layer_call_fn_480383Й\]^_MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ †
7__inference_batch_normalization_20_layer_call_fn_480445e\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ " К€€€€€€€€€22 †
7__inference_batch_normalization_20_layer_call_fn_480458e\]^_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ " К€€€€€€€€€22 »
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480517rklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ »
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480535rklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ "-Ґ*
#К 
0€€€€€€€€€22 
Ъ н
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480592ЦklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ н
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_480610ЦklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ †
7__inference_batch_normalization_21_layer_call_fn_480548eklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p
™ " К€€€€€€€€€22 †
7__inference_batch_normalization_21_layer_call_fn_480561eklmn;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22 
p 
™ " К€€€€€€€€€22 ≈
7__inference_batch_normalization_21_layer_call_fn_480623ЙklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ≈
7__inference_batch_normalization_21_layer_call_fn_480636ЙklmnMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ с
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480733ЪИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ с
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480751ЪИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ћ
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480808vИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ ћ
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_480826vИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ …
7__inference_batch_normalization_22_layer_call_fn_480764НИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@…
7__inference_batch_normalization_22_layer_call_fn_480777НИЙКЛMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@§
7__inference_batch_normalization_22_layer_call_fn_480839iИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ " К€€€€€€€€€22@§
7__inference_batch_normalization_22_layer_call_fn_480852iИЙКЛ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ " К€€€€€€€€€22@с
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480911ЪЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ с
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480929ЪЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ћ
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_480986vЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ ћ
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_481004vЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ "-Ґ*
#К 
0€€€€€€€€€22@
Ъ …
7__inference_batch_normalization_23_layer_call_fn_480942НЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@…
7__inference_batch_normalization_23_layer_call_fn_480955НЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@§
7__inference_batch_normalization_23_layer_call_fn_481017iЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p
™ " К€€€€€€€€€22@§
7__inference_batch_normalization_23_layer_call_fn_481030iЧШЩЪ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€22@
p 
™ " К€€€€€€€€€22@Џ
E__inference_conv2d_27_layer_call_and_return_conditional_losses_475188Р#$IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≤
*__inference_conv2d_27_layer_call_fn_475198Г#$IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Џ
E__inference_conv2d_28_layer_call_and_return_conditional_losses_475226Р)*IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≤
*__inference_conv2d_28_layer_call_fn_475236Г)*IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Џ
E__inference_conv2d_29_layer_call_and_return_conditional_losses_475389Р89IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≤
*__inference_conv2d_29_layer_call_fn_475399Г89IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Џ
E__inference_conv2d_30_layer_call_and_return_conditional_losses_475553РOPIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≤
*__inference_conv2d_30_layer_call_fn_475563ГOPIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Џ
E__inference_conv2d_31_layer_call_and_return_conditional_losses_475591РUVIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≤
*__inference_conv2d_31_layer_call_fn_475601ГUVIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Џ
E__inference_conv2d_32_layer_call_and_return_conditional_losses_475754РdeIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≤
*__inference_conv2d_32_layer_call_fn_475764ГdeIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Џ
E__inference_conv2d_33_layer_call_and_return_conditional_losses_475918Р{|IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≤
*__inference_conv2d_33_layer_call_fn_475928Г{|IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@№
E__inference_conv2d_34_layer_call_and_return_conditional_losses_475956ТБВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
*__inference_conv2d_34_layer_call_fn_475966ЕБВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@№
E__inference_conv2d_35_layer_call_and_return_conditional_losses_476119ТРСIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ і
*__inference_conv2d_35_layer_call_fn_476129ЕРСIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@®
D__inference_dense_10_layer_call_and_return_conditional_losses_481185`єЇ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
)__inference_dense_10_layer_call_fn_481194SєЇ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€АІ
D__inference_dense_11_layer_call_and_return_conditional_losses_481264_√ƒ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
)__inference_dense_11_layer_call_fn_481273R√ƒ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€¶
C__inference_dense_9_layer_call_and_return_conditional_losses_481106_ѓ∞/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ~
(__inference_dense_9_layer_call_fn_481115Rѓ∞/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€АІ
E__inference_dropout_6_layer_call_and_return_conditional_losses_481127^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ І
E__inference_dropout_6_layer_call_and_return_conditional_losses_481132^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dropout_6_layer_call_fn_481137Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€А
*__inference_dropout_6_layer_call_fn_481142Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€АІ
E__inference_dropout_7_layer_call_and_return_conditional_losses_481206^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ І
E__inference_dropout_7_layer_call_and_return_conditional_losses_481211^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
*__inference_dropout_7_layer_call_fn_481216Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€А
*__inference_dropout_7_layer_call_fn_481221Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А°
E__inference_flatten_3_layer_call_and_return_conditional_losses_481058X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ y
*__inference_flatten_3_layer_call_fn_481063K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@я
V__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_476262ДRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ ґ
;__inference_global_average_pooling2d_3_layer_call_fn_476268wRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€;
__inference_loss_fn_0_481286#Ґ

Ґ 
™ "К <
__inference_loss_fn_10_481416dҐ

Ґ 
™ "К <
__inference_loss_fn_11_481429eҐ

Ґ 
™ "К <
__inference_loss_fn_12_481442{Ґ

Ґ 
™ "К <
__inference_loss_fn_13_481455|Ґ

Ґ 
™ "К =
__inference_loss_fn_14_481468БҐ

Ґ 
™ "К =
__inference_loss_fn_15_481481ВҐ

Ґ 
™ "К =
__inference_loss_fn_16_481494РҐ

Ґ 
™ "К =
__inference_loss_fn_17_481507СҐ

Ґ 
™ "К =
__inference_loss_fn_18_481520ѓҐ

Ґ 
™ "К =
__inference_loss_fn_19_481533∞Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_481299$Ґ

Ґ 
™ "К =
__inference_loss_fn_20_481546єҐ

Ґ 
™ "К =
__inference_loss_fn_21_481559ЇҐ

Ґ 
™ "К =
__inference_loss_fn_22_481572√Ґ

Ґ 
™ "К =
__inference_loss_fn_23_481585ƒҐ

Ґ 
™ "К ;
__inference_loss_fn_2_481312)Ґ

Ґ 
™ "К ;
__inference_loss_fn_3_481325*Ґ

Ґ 
™ "К ;
__inference_loss_fn_4_4813388Ґ

Ґ 
™ "К ;
__inference_loss_fn_5_4813519Ґ

Ґ 
™ "К ;
__inference_loss_fn_6_481364OҐ

Ґ 
™ "К ;
__inference_loss_fn_7_481377PҐ

Ґ 
™ "К ;
__inference_loss_fn_8_481390UҐ

Ґ 
™ "К ;
__inference_loss_fn_9_481403VҐ

Ґ 
™ "К х
C__inference_model_3_layer_call_and_return_conditional_losses_477302≠B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_4€€€€€€€€€22
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ х
C__inference_model_3_layer_call_and_return_conditional_losses_477622≠B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_4€€€€€€€€€22
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ф
C__inference_model_3_layer_call_and_return_conditional_losses_479296ђB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ф
C__inference_model_3_layer_call_and_return_conditional_losses_479668ђB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ќ
(__inference_model_3_layer_call_fn_478044†B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_4€€€€€€€€€22
p

 
™ "К€€€€€€€€€Ќ
(__inference_model_3_layer_call_fn_478465†B#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ@Ґ=
6Ґ3
)К&
input_4€€€€€€€€€22
p 

 
™ "К€€€€€€€€€ћ
(__inference_model_3_layer_call_fn_479769ЯB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p

 
™ "К€€€€€€€€€ћ
(__inference_model_3_layer_call_fn_479870ЯB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€22
p 

 
™ "К€€€€€€€€€з
$__inference_signature_wrapper_478832ЊB#$)*012389?@ABOPUV\]^_deklmn{|БВИЙКЛРСЧШЩЪѓ∞єЇ√ƒCҐ@
Ґ 
9™6
4
input_4)К&
input_4€€€€€€€€€22"3™0
.
dense_11"К
dense_11€€€€€€€€€