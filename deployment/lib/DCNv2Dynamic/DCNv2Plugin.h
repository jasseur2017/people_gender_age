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
// #include "NvInferRuntime.h"
#include "NvInfer.h"
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
		};

		constexpr int N_DIM = 0;
		constexpr int C_DIM = 1;
		constexpr int H_DIM = 2;
		constexpr int W_DIM = 3;

		void enqueue_call(
			const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream,
			const Dims& input_shape, const Dims& offset_shape, const Dims& mask_shape, const Dims& output_shape,
			DataType mType, cublasHandle_t cublasHandle_
		);

		inline unsigned int getElementSize(DataType t);

		class DCNv2DynamicPlugin : public IPluginV2DynamicExt
		{
		public:
			DCNv2DynamicPlugin();

			DCNv2DynamicPlugin(const void* data, size_t length, const std::string& name);
			
			DCNv2DynamicPlugin(DCNv2Parameters param, const std::string& name);
			
			~DCNv2DynamicPlugin() override;

			// IPluginV2DynamicExt Methods
			IPluginV2DynamicExt* clone() const noexcept override;
			
			DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs,
				int nbInputs, IExprBuilder& exprBuilder) noexcept override;
			
			bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
				int nbInputs, int nbOutputs) noexcept override;
			
			void configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
				const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept override;

			size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
				const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
			
			int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
				const void* const* inputs, void* const* outputs, void* workspace,
				cudaStream_t stream) noexcept override;

			// IPluginV2Ext Methods
			DataType getOutputDataType(int index, const DataType* inputTypes,
				int nbInputs) const noexcept override;

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
			cublasHandle_t cublasHandle_;
			cudnnHandle_t cudnnHandle_;

			DataType mType;
			Dims input_shape;
			Dims offset_shape;
			Dims mask_shape;
			Dims output_shape;
			DCNv2Parameters mParam;
			//Weights mDeviceWeights, mDeviceBiases;

		// public:
		// 	using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
		// 	using nvinfer1::IPluginV2DynamicExt::configurePlugin;
		// 	using nvinfer1::IPluginV2DynamicExt::enqueue;
		// 	using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
		// 	using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
		// 	using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
		// 	using nvinfer1::IPluginV2DynamicExt::supportsFormat;
		}; // class DCNv2Plugin

		class DCNv2DynamicPluginCreator : public IPluginCreator
		{
		public:
			DCNv2DynamicPluginCreator();

			const char* getPluginName() const noexcept override;

			const char* getPluginVersion() const noexcept override;

			const PluginFieldCollection* getFieldNames() noexcept override;

			IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

			IPluginV2* deserializePlugin(
				const char* name, const void* serialData, size_t serialLength
			) noexcept override;

			void setPluginNamespace(const char* pluginNamespace) noexcept override;

			const char* getPluginNamespace() const noexcept override;

		private:
			static PluginFieldCollection mFC;
			static std::vector<PluginField> mPluginAttributes;

			std::string mNamespace;
		}; // class DCNv2DynamicPluginCreator
	} // namespace plugin
} //namespace nvinfer1

#endif // TRT_DCNV2_PLUGIN_H
