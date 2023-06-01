/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DCNv2Plugin.h"
// #include "NvInfer.h"
// #include "serialize.hpp"
#include <cassert>
// #include <cstring>
#include <iostream>
#include <vector>
// #include <cublas_v2.h>
// #include <cudnn.h>
// #include <sstream>
// #include <cuda.h>

namespace nvinfer1
{
namespace dcnv2
{
namespace
{
	static const char* DCNV2_NAME{"deform_conv2d"};
	static const char* DCNV2_VERSION{"1"};
} //namespace

// Static class fields initialization
PluginFieldCollection DCNv2DynamicPluginCreator::mFC{};
std::vector<PluginField> DCNv2DynamicPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DCNv2DynamicPluginCreator);

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

DCNv2DynamicPlugin::DCNv2DynamicPlugin(const void* data, size_t length,
	const std::string& name)
	: mLayerName(name)
{
	const char* d = reinterpret_cast<const char*>(data), *a = d;

	mParam.stride_h = read<int>(d);
	mParam.stride_w = read<int>(d);
	mParam.pad_h = read<int>(d);
	mParam.pad_w = read<int>(d);
	mParam.dil_h = read<int>(d);
	mParam.dil_w = read<int>(d);
	mType = read<DataType>(d);
	input_shape = read<Dims>(d);
	offset_shape = read<Dims>(d);
	mask_shape = read<Dims>(d);
	output_shape = read<Dims>(d);
}

DCNv2DynamicPlugin::DCNv2DynamicPlugin(DCNv2Parameters param, const std::string& name)
	: mParam{param}, mLayerName(name)
{
}

DCNv2DynamicPlugin::~DCNv2DynamicPlugin(){}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* DCNv2DynamicPlugin::clone() const noexcept
{
	auto p = new DCNv2DynamicPlugin(mParam, mLayerName);
	p->setPluginNamespace(mNamespace.c_str());

	return p;
}

DimsExprs DCNv2DynamicPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs,
	int nbInputs, IExprBuilder& exprBuilder) noexcept
{
	// Validate input arguments
	assert(nbInputs == 5);

	DimsExprs ret;
	ret.nbDims = 4;
	ret.d[0] = inputs[0].d[0];
	ret.d[1] = inputs[1].d[0];
	ret.d[2] = inputs[0].d[2];
	ret.d[3] = inputs[0].d[3];
	return ret;
}

bool DCNv2DynamicPlugin::supportsFormatCombination(
	int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs
) noexcept {
	// Validate input arguments
	assert(nbInputs == 5);
	assert(nbOutputs == 1);
	assert(pos < nbInputs + nbOutputs);

	const PluginTensorDesc& desc = inOut[pos];
	return ((desc.type == DataType::kFLOAT || desc.type == DataType::kHALF)
		&& desc.format == TensorFormat::kLINEAR //kNCHW
		&& desc.type == inOut[0].type);
}

void DCNv2DynamicPlugin::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
	const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
	// Validate input arguments
	assert(nbInputs == 5);
	assert(nbOutputs == 1);

	mType = inputs[0].desc.type;
	input_shape = inputs[0].desc.dims;
	offset_shape = inputs[2].desc.dims;
	mask_shape = inputs[3].desc.dims;
	output_shape = outputs[0].desc.dims;
}

size_t DCNv2DynamicPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
	const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
	size_t im2colSize = static_cast<size_t>(
		input_shape.d[C_DIM] * mask_shape.d[C_DIM] * output_shape.d[H_DIM] * output_shape.d[W_DIM]
	);

	int maxBatchSize = 1;
	return im2colSize * maxBatchSize * (mType == DataType::kFLOAT ? 4 : 2);
}

int DCNv2DynamicPlugin::enqueue(const PluginTensorDesc* inputDesc,
	const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs,
	void* workspace, cudaStream_t stream) noexcept
{
	enqueue_call(
		inputs, outputs, workspace, stream,
		input_shape, offset_shape, mask_shape, output_shape,
		mType, cublasHandle_
	);		
}

// IPluginV2Ext Methods
DataType DCNv2DynamicPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
	assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
	return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin
// the access to some context resource
void DCNv2DynamicPlugin::attachToContext(cudnnContext* cudnnContext,
	cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
	cublasHandle_ = cublasContext;
	cudnnHandle_ = cudnnContext;
}

// Detach the plugin object from its execution context
void DCNv2DynamicPlugin::detachFromContext() noexcept {}

// IPluginV2 Methods
const char* DCNv2DynamicPlugin::getPluginType() const noexcept
{
	return DCNV2_NAME;
}

const char* DCNv2DynamicPlugin::getPluginVersion() const noexcept
{
	return DCNV2_VERSION;
}

int DCNv2DynamicPlugin::getNbOutputs() const noexcept
{
	return 1;
}

int DCNv2DynamicPlugin::initialize() noexcept
{
	return 0;
}

void DCNv2DynamicPlugin::terminate() noexcept {}

size_t DCNv2DynamicPlugin::getSerializationSize() const noexcept
{
	return sizeof(DCNv2Parameters) + sizeof(DataType) + 4 * sizeof(Dims);
}

void DCNv2DynamicPlugin::serialize(void* buffer) const noexcept
{
	char* d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mParam.stride_h);
	write(d, mParam.stride_w);
	write(d, mParam.pad_h);
	write(d, mParam.pad_w);
	write(d, mParam.dil_h);
	write(d, mParam.dil_w);
	write(d, mType);
	write(d, input_shape);
	write(d, offset_shape);
	write(d, mask_shape);
	write(d, output_shape);
}

void DCNv2DynamicPlugin::destroy() noexcept
{
	delete this;
}

void DCNv2DynamicPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
	mNamespace = pluginNamespace;
}

const char* DCNv2DynamicPlugin::getPluginNamespace() const noexcept
{
	return mNamespace.c_str();
}

////////////////////
DCNv2DynamicPluginCreator::DCNv2DynamicPluginCreator()
{
	mPluginAttributes.emplace_back(PluginField("stride_h", nullptr,
		PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("stride_w", nullptr,
		PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("pad_h", nullptr,
		PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("pad_w", nullptr,
		PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("dil_h", nullptr,
		PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("dil_w", nullptr,
		PluginFieldType::kINT32, 1));
	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* DCNv2DynamicPluginCreator::getPluginName() const noexcept
{
	return DCNV2_NAME;
}

const char* DCNv2DynamicPluginCreator::getPluginVersion() const noexcept
{
	return DCNV2_VERSION;
}

const PluginFieldCollection* DCNv2DynamicPluginCreator::getFieldNames() noexcept
{
	return &mFC;
}

IPluginV2* DCNv2DynamicPluginCreator::createPlugin(
	const char* name, const PluginFieldCollection* fc
) noexcept {
	DCNv2Parameters dcnv2Params;
	const PluginField* fields = fc->fields;

	for (int i = 0 ; i < fc->nbFields ; i++)
	{
		const char* attrName = fields[i].name;
		assert(fields[i].type == PluginFieldType::kINT32);
		int d = *static_cast<const int*>(fields[i].data);	
		if (!strcmp(attrName, "stride_h"))
		{
			dcnv2Params.stride_h = d;
		}
		else if (!strcmp(attrName, "stride_w"))
		{
			dcnv2Params.stride_w = d;
		}
		else if (!strcmp(attrName, "pad_h"))
		{
			dcnv2Params.pad_h = d;
		}
		else if (!strcmp(attrName, "pad_w"))
		{
			dcnv2Params.pad_w = d;
		}
		else if (!strcmp(attrName, "dil_h"))
		{
			dcnv2Params.dil_h = d;
		}
		else if (!strcmp(attrName, "dil_w"))
		{
			dcnv2Params.dil_w = d;
		}
	}

	DCNv2DynamicPlugin* p = new DCNv2DynamicPlugin(dcnv2Params, name);
	return p;
}

IPluginV2* DCNv2DynamicPluginCreator::deserializePlugin(
	const char* name, const void* serialData, size_t serialLength
) noexcept {
	// This object will be deleted when the network is destroyed, which will
	// call DCNv2DynamicPlugin::destroy()
	return new DCNv2DynamicPlugin(serialData, serialLength, name);
}

void DCNv2DynamicPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
	mNamespace = pluginNamespace;
}

const char* DCNv2DynamicPluginCreator::getPluginNamespace() const noexcept
{
	return mNamespace.c_str();
}

inline unsigned int getElementSize(DataType t)
{
	switch (t)
	{
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;	
}

} // namespace dcnv2
} // namespace nvinfer1
