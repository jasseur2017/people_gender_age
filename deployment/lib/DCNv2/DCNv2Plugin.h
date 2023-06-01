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

#ifndef TRT_DCNV2_PLUGIN_H
#define TRT_DCNV2_PLUGIN_H

// #include "NvInferPlugin.h"
#include "NvInfer.h"
// #include "NvInferRuntime.h"
// #include "kernel.h"
// #include "plugin.h"
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1
{
	namespace dcnv2
	{
		struct DCNv2Parameters
		{
			int stride_h;
			int stride_w;
			int pad_h;
			int pad_w;
			int dil_h;
			int dil_w;
			int out_channels;
		};

		constexpr int C_DIM = 0;
		constexpr int H_DIM = 1;
		constexpr int W_DIM = 2;

		void enqueue_call(
			int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
			const Dims& input_shape, const Dims& offset_shape, const Dims& mask_shape, const Dims& output_shape,
			DataType mType, cublasHandle_t mCublasHandle
		);

		inline unsigned int getElementSize(DataType t);

		class DCNv2Plugin : public IPluginV2Ext
		{
		public:
			DCNv2Plugin();

			DCNv2Plugin(const void* data, size_t length, const std::string& name);
			
			DCNv2Plugin(DCNv2Parameters param, const std::string& name);
			
			~DCNv2Plugin() override;

			// IPluginV2Ext Methods
			IPluginV2Ext* clone() const noexcept override;
			
			Dims getOutputDimensions(int outputIndex, const Dims* inputs, int nbInputs) noexcept override;
			
			bool supportsFormat(DataType type, PluginFormat format) const noexcept override;	

			void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
				const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
				const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

			size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

			int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
				void* workspace, cudaStream_t stream) noexcept override;

			DataType getOutputDataType(int index, const DataType* inputTypes,
				int nbInputs) const noexcept override;

			bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

			bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

			void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
				IGpuAllocator* gpuAllocator) noexcept override;

			void detachFromContext() noexcept override;

			// IPluginV2 Methods
			const char* getPluginType() const noexcept override;

			const char* getPluginVersion() const noexcept override;

			int getNbOutputs() const noexcept override;

			int initialize() noexcept override;

			void terminate() noexcept override;

			size_t getSerializationSize() const noexcept override;

			void serialize(void* buffer) const noexcept override;

			void destroy() noexcept override;

			void setPluginNamespace(const char* pluginNamespace) noexcept override;

			const char* getPluginNamespace() const noexcept override;

		private:
			const std::string mLayerName;
			std::string mNamespace;
			cublasHandle_t mCublasHandle;
			cudnnHandle_t mCudnnHandle;

			DataType mType;
			Dims input_shape;
			Dims offset_shape;
			Dims mask_shape;
			Dims output_shape;
			DCNv2Parameters mParam;
			//Weights mDeviceWeights, mDeviceBiases;

		}; // class DCNv2Plugin

		class DCNv2PluginCreator : public IPluginCreator
		{
		public:
			DCNv2PluginCreator();

			const char* getPluginName() const noexcept override;

			const char* getPluginVersion() const noexcept override;

			const PluginFieldCollection* getFieldNames() noexcept override;

			IPluginV2* createPlugin(const char* name,
				const PluginFieldCollection* fc) noexcept override;

			IPluginV2* deserializePlugin(const char* name, const void* serialData, 
				size_t serialLength) noexcept override;

			void setPluginNamespace(const char* pluginNamespace) noexcept override;

			const char* getPluginNamespace() const noexcept override;

		private:
			static PluginFieldCollection mFC;
			static std::vector<PluginField> mPluginAttributes;
			std::string mNamespace;
		}; // class DCNv2PluginCreator
	} // namespace plugin
} //namespace nvinfer1

#endif // TRT_DCNV2_PLUGIN_H
