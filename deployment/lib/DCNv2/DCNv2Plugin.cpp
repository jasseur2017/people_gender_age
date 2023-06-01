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
PluginFieldCollection DCNv2PluginCreator::mFC{};
std::vector<PluginField> DCNv2PluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);

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

DCNv2Plugin::DCNv2Plugin(const void* data, size_t length,
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
	mParam.out_channels = read<int>(d);
	mType = read<DataType>(d);
	input_shape = read<Dims>(d);
	offset_shape = read<Dims>(d);
	mask_shape = read<Dims>(d);
	output_shape = read<Dims>(d);
}

DCNv2Plugin::DCNv2Plugin(DCNv2Parameters param, const std::string& name)
	: mParam{param}, mLayerName(name)
{
}

DCNv2Plugin::~DCNv2Plugin(){}

// IPluginV2Ext Methods
IPluginV2Ext* DCNv2Plugin::clone() const noexcept
{
	auto p = new DCNv2Plugin(*this);
	p->setPluginNamespace(mNamespace.c_str());
	return p;
}

Dims DCNv2Plugin::getOutputDimensions(int outputIndex, const Dims* inputs, int nbInputs) noexcept
{
	// Validate input arguments
	assert(nbInputs == 5);
	// return DimsCHW(mParam.out_channels, inputs[0].d[H_DIM], inputs[0].d[W_DIM]);
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = 3;
    dimsOutput.d[0] = mParam.out_channels;
    dimsOutput.d[1] = inputs[0].d[H_DIM];
    dimsOutput.d[2] = inputs[0].d[W_DIM];
    return dimsOutput;
}

bool DCNv2Plugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
	return ((type == DataType::kFLOAT || type == DataType::kHALF)
				&& format == PluginFormat::kLINEAR);
}

void DCNv2Plugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
	const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
	const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
	// Validate input arguments
	assert(nbInputs == 5);
	assert(nbOutputs == 1);

	mType = inputTypes[0];
	input_shape = inputDims[0];
	offset_shape = inputDims[2];
	mask_shape = inputDims[3];
	output_shape = outputDims[0];
}

size_t DCNv2Plugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
	size_t im2colSize = static_cast<size_t>(
		input_shape.d[C_DIM] * mask_shape.d[C_DIM] * output_shape.d[H_DIM] * output_shape.d[W_DIM]
	);

	return im2colSize * maxBatchSize * (mType == DataType::kFLOAT ? 4 : 2);
}

int DCNv2Plugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs,
	void* workspace, cudaStream_t stream) noexcept
{
	enqueue_call(
		batchSize, inputs, outputs, workspace, stream,
		input_shape, offset_shape, mask_shape, output_shape,
		mType, mCublasHandle
	);
}

DataType DCNv2Plugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
	assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
	return inputTypes[0];
}

bool DCNv2Plugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
	return false;
}

bool DCNv2Plugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
	return false;
}

// Attach the plugin object to an execution context and grant the plugin
// the access to some context resource
void DCNv2Plugin::attachToContext(cudnnContext* cudnnContext,
	cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
	mCublasHandle = cublasContext;
	mCudnnHandle = cudnnContext;
}

// Detach the plugin object from its execution context
void DCNv2Plugin::detachFromContext() noexcept {}

// IPluginV2 Methods
const char* DCNv2Plugin::getPluginType() const noexcept
{
	return DCNV2_NAME;
}

const char* DCNv2Plugin::getPluginVersion() const noexcept
{
	return DCNV2_VERSION;
}

int DCNv2Plugin::getNbOutputs() const noexcept
{
	return 1;
}

int DCNv2Plugin::initialize() noexcept
{
	return 0;
}

void DCNv2Plugin::terminate() noexcept {}

size_t DCNv2Plugin::getSerializationSize() const noexcept
{
	return sizeof(DCNv2Parameters) + sizeof(DataType) + 4 * sizeof(Dims);
}

void DCNv2Plugin::serialize(void* buffer) const noexcept
{
	char* d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, mParam.stride_h);
	write(d, mParam.stride_w);
	write(d, mParam.pad_h);
	write(d, mParam.pad_w);
	write(d, mParam.dil_h);
	write(d, mParam.dil_w);
	write(d, mParam.out_channels);
	write(d, mType);
	write(d, input_shape);
	write(d, offset_shape);
	write(d, mask_shape);
	write(d, output_shape);	
}

void DCNv2Plugin::destroy() noexcept
{
	delete this;
}

void DCNv2Plugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
	mNamespace = pluginNamespace;
}

const char* DCNv2Plugin::getPluginNamespace() const noexcept
{
	return mNamespace.c_str();
}

////////////////////
DCNv2PluginCreator::DCNv2PluginCreator()
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
	mPluginAttributes.emplace_back(PluginField("out_channels", nullptr,
		PluginFieldType::kINT32, 1));
	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* DCNv2PluginCreator::getPluginName() const noexcept
{
	return DCNV2_NAME;
}

const char* DCNv2PluginCreator::getPluginVersion() const noexcept
{
	return DCNV2_VERSION;
}

const PluginFieldCollection* DCNv2PluginCreator::getFieldNames() noexcept
{
	return &mFC;
}

IPluginV2* DCNv2PluginCreator::createPlugin(const char* name,
	const PluginFieldCollection* fc) noexcept
{
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
		else if (!strcmp(attrName, "out_channels"))
		{
			dcnv2Params.out_channels = d;
		}
	}

	DCNv2Plugin* p = new DCNv2Plugin(dcnv2Params, name);
	return p;
}

IPluginV2* DCNv2PluginCreator::deserializePlugin(const char* name,
	const void* serialData, size_t serialLength) noexcept
{
	// This object will be deleted when the network is destroyed, which will
	// call DCNv2Plugin::destroy()
	return new DCNv2Plugin(serialData, serialLength, name);
}

void DCNv2PluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
	mNamespace = pluginNamespace;
}

const char* DCNv2PluginCreator::getPluginNamespace() const noexcept
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
