# TensorFlow Serving源码阅读

## 源码目录结构

TensorFlow Serving使用C++编写，bazel进行构建管理，提供gRpc和Restfull两种接口方式访问服务。tensorflow_serving是TensorFlow Serving部分的源码，third_party是第三方扩展库的源码，tools下面是运维的相关脚本，推荐使用docker容器化部署。官方文档上介绍是高弹性的，高性能的用于部署机器学习模型的服务。

TensorFlow Serving主要依赖的是Tensorflow和libevent 。libevent是C语言开发的高性能事件处理框架跨平台，可用于开发http的服务器，目前Google的大部分Http服务器，包括Chrome浏览器都是基于libevent开发的。Tensorflow Serving属于Tensorflow家族列表中产品，源码编译依赖Tensorflow。

TensorFlow Serving的架构如下图，其实我们可以简单的认为TensorFlow Serving是一个TensorFlow模型的计算能力发布的容器，它架构非常的抽象，其实是可以扩展到其它种类的模型，可以参考官网的连接[https://tensorflow.google.cn/tfx/serving/custom_servable]。

![serving_architecture](/home/yczhang/Desktop/tensorflow/serving-2.1.0/serving_architecture.svg)


1. TensorFlow Serving使用Servable来封装可服务的对象，典型的可服务对象有SavedModelBundle (`tensorflow::Session`)。Tensorflow中推荐使用SavedModel格式保存模型，以便使用Predict API加载和服务；

2. 构建CPU优化服务二进制，按照Tensoflow的文档，官方建议从源码编译TensorFlow和TensorFlow Serving，因为源码中针对CPU和GPU的具体类型优化非常多，能够基于运行二进制文件的主机平台的CPU上使用所有可用的优化；

3. 提高客户端的效率，使用专有的硬件来加速计算过程，无需客户端加载TensorFlow；


## 源码分析

### 实现预测源码分析

tensorflow_serving/model_servers目录是服务的实现模块。
```bazel
cc_binary(
    name = "tensorflow_model_server",
    stamp = 1,
    visibility = [
        ":testing",
        "//tensorflow_serving:internal",
    ],
    deps = [
        ":tensorflow_model_server_main_lib",
    ],
)
```
文件tensorflow_serving/model_servers/server.h定义的是服务的组合，包括：

```c++
  std::unique_ptr<::grpc::Server> grpc_server_;
  std::unique_ptr<net_http::HTTPServerInterface> http_server_;
```
重点看下http服务的实现。net_http包是Google封装的http服务的包，基于libevent进行封装的，并且实现了一个http服务的框架。
工厂方法创建HTTPServerInterface，其中实现者是EvHTTPServer，基于上面说的高性能的libevent开发的。

```c++
std::unique_ptr<net_http::HTTPServerInterface> CreateAndStartHttpServer(
    int port, int num_threads, int timeout_in_ms,
    const MonitoringConfig& monitoring_config, ServerCore* core) {
  auto options = absl::make_unique<net_http::ServerOptions>();
  options->AddPort(static_cast<uint32_t>(port));
  options->SetExecutor(absl::make_unique<RequestExecutor>(num_threads));

  auto server = net_http::CreateEvHTTPServer(std::move(options));
  if (server == nullptr) {
    return nullptr;
  }

  // Register handler for prometheus metric endpoint.
  if (monitoring_config.prometheus_config().enable()) {
    std::shared_ptr<PrometheusExporter> exporter =
        std::make_shared<PrometheusExporter>();
    net_http::RequestHandlerOptions prometheus_request_options;
    PrometheusConfig prometheus_config = monitoring_config.prometheus_config();
    auto path = prometheus_config.path().empty()
                    ? PrometheusExporter::kPrometheusPath
                    : prometheus_config.path();
    server->RegisterRequestHandler(
        path,
        [exporter, path](net_http::ServerRequestInterface* req) {
          ProcessPrometheusRequest(exporter.get(), path, req);
        },
        prometheus_request_options);
  }

  std::shared_ptr<RestApiRequestDispatcher> dispatcher =
      std::make_shared<RestApiRequestDispatcher>(timeout_in_ms, core);
  net_http::RequestHandlerOptions handler_options;
  server->RegisterRequestDispatcher(
      [dispatcher](net_http::ServerRequestInterface* req) {
        return dispatcher->Dispatch(req);
      },
      handler_options);
  if (server->StartAcceptingRequests()) {
    return server;
  }
  return nullptr;
}
```
最重要的两个类：
1. RequestExecutor是执行器，多线程并发。
2. RestApiRequestDispatcher派发器。

RestApiRequestDispatcher内部是正则表达式语法的一个处理器，使用正则表达来提取url，包括：model_name、model_version。
该类下面的函数是提供的全部Restful接口的定义。

```c++
class HttpRestApiHandler {
 public:
  // Returns a regex that captures all API paths handled by this handler.
  // Typical use of this method is to register request paths with underlying
  // HTTP server, so incoming requests can be forwarded to this handler.
  static const char* const kPathRegex;

  // API calls are configured to timeout after `run_optons.timeout_in_ms`.
  // `core` is not owned and is expected to outlive HttpRestApiHandler
  // instance.
  HttpRestApiHandler(const RunOptions& run_options, ServerCore* core);

  ~HttpRestApiHandler();

  // Process a HTTP request.
  //
  // In case of errors, the `headers` and `output` are still relevant as they
  // contain detailed error messages, that can be relayed back to the client.
  Status ProcessRequest(const absl::string_view http_method,
                        const absl::string_view request_path,
                        const absl::string_view request_body,
                        std::vector<std::pair<string, string>>* headers,
                        string* output);

 private:
  Status ProcessClassifyRequest(const absl::string_view model_name,
                                const absl::optional<int64>& model_version,
                                const absl::string_view request_body,
                                string* output);
  Status ProcessRegressRequest(const absl::string_view model_name,
                               const absl::optional<int64>& model_version,
                               const absl::string_view request_body,
                               string* output);
  Status ProcessPredictRequest(const absl::string_view model_name,
                               const absl::optional<int64>& model_version,
                               const absl::string_view request_body,
                               string* output);
  Status ProcessModelStatusRequest(const absl::string_view model_name,
                                   const absl::string_view model_version_str,
                                   string* output);
  Status ProcessModelMetadataRequest(const absl::string_view model_name,
                                     const absl::string_view model_version_str,
                                     string* output);
  Status GetInfoMap(const ModelSpec& model_spec, const string& signature_name,
                    ::google::protobuf::Map<string, tensorflow::TensorInfo>* infomap);

  const RunOptions run_options_;
  ServerCore* core_;
  std::unique_ptr<TensorflowPredictor> predictor_;
  const RE2 prediction_api_regex_;
  const RE2 modelstatus_api_regex_;
};

```

```c++
Status TensorflowPredictor::PredictWithModelSpec(const RunOptions& run_options,
                                                 ServerCore* core,
                                                 const ModelSpec& model_spec,
                                                 const PredictRequest& request,
                                                 PredictResponse* response) {
  if (use_saved_model_) {
    ServableHandle<SavedModelBundle> bundle;
    TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
    return internal::RunPredict(
        run_options, bundle->meta_graph_def, bundle.id().version,
        core->predict_response_tensor_serialization_option(),
        bundle->session.get(), request, response);
  }
  ServableHandle<SessionBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  // SessionBundle is officially deprecated. SessionBundlePredict is for
  // backward compatibility.
  return SessionBundlePredict(
      run_options, bundle->meta_graph_def, bundle.id().version,
      core->predict_response_tensor_serialization_option(), request, response,
      bundle->session.get());
}
```
默认是会对输入的所有的Tensor打包批量来执行预测，提高效率批量执行的效率，使用C++提供的接口优势。注意ServableHandle<SessionBundle> bundle;这个bundle需要及时释放，避免占用导致新资源的更新和加载不及时。

以预测为例，会最终通过下面这行调用到tensorflow内部执行具体的计算。
```c++
session->Run(run_options, input_tensors,
                                  output_tensor_names, {}, &outputs,
                                  &run_metadata)
```

### 模型加载过程

ServerCore是服务的核心类，位于tensorflow_serving/model_servers包下面。

ServerCore实现Manager接口，Manager管理所有的提供服务能力对象以及它们的生命周期，Manager接口定义如下：

```c++
/// Manager is responsible for loading, unloading, lookup and lifetime
/// management of all Servable objects via their Loaders.
class Manager {
 public:
  virtual ~Manager() = default;

  /// Gets a list of all available servable ids, i.e. each of these can
  /// be retrieved using GetServableHandle.
  virtual std::vector<ServableId> ListAvailableServableIds() const = 0;

  /// Returns a map of all the currently available servables of a particular
  /// type T. The map is from the servable's id to its corresponding handle.
  ///
  /// IMPORTANT: The caller should not hold onto the handles for a long time,
  /// because holding them will delay servable loading and unloading.
  template <typename T>
  std::map<ServableId, ServableHandle<T>> GetAvailableServableHandles() const;

  /// Returns a ServableHandle given a ServableRequest. Returns error if no such
  /// Servable is available -- e.g. not yet loaded, has been quiesced/unloaded,
  /// etc. Callers may assume that an OK status indicates a non-null handle.
  ///
  /// IMPORTANT: The caller should not hold onto the handles for a long time,
  /// because holding them will delay servable loading and unloading.
  template <typename T>
  Status GetServableHandle(const ServableRequest& request,
                           ServableHandle<T>* const handle);

 private:
  friend class ManagerWrapper;

  // Returns an UntypedServableHandle given a ServableRequest.
  // Returns error if no such Servable is available -- e.g. not yet loaded, has
  // been quiesced/unloaded, etc.
  virtual Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) = 0;

  // Returns a map of all the available servable ids to their corresponding
  // UntypedServableHandles.
  virtual std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const = 0;
};
```

服务启动时首先调用CreateAdapters(&adapters)创建adapters。然后调用CreateRouter(routes, &adapters, &router)创建router。接着调用CreateStoragePathSource(source_config, router.get(), &source)创建source。最后ConnectAdaptersToManagerAndAwaitModelLoads(&adapters)，将adapters连接到管理者。这里的管理者是AspiredVersionsManager：

```c++
class AspiredVersionsManager : public Manager,
                               public Target<std::unique_ptr<Loader>> 
```

AspiredVersionsManager是ServerCore内部另外一个管理者，可以看到AspiredVersionsManager既是管理者，也是Target，是Target表示可以接受Source的消息。是管理者代表具备管理具有服务能力的对对象。

```c++
Status ServerCore::AddModelsViaModelConfigList() {
  const bool is_first_config = storage_path_source_and_router_ == nullopt;

  // Create/reload the source, source router and source adapters.
  const FileSystemStoragePathSourceConfig source_config =
      CreateStoragePathSourceConfig(config_);
  DynamicSourceRouter<StoragePath>::Routes routes;
  TF_RETURN_IF_ERROR(CreateStoragePathRoutes(config_, &routes));
  if (is_first_config) {
    // Construct the following source topology:
    //   Source -> Router -> Adapter_0 (for models using platform 0)
    //                    -> Adapter_1 (for models using platform 1)
    //                    -> ...
    //                    -> ErrorAdapter (for unrecognized models)
    SourceAdapters adapters;
    TF_RETURN_IF_ERROR(CreateAdapters(&adapters));
    std::unique_ptr<DynamicSourceRouter<StoragePath>> router;
    TF_RETURN_IF_ERROR(CreateRouter(routes, &adapters, &router));
    std::unique_ptr<FileSystemStoragePathSource> source;
    TF_RETURN_IF_ERROR(
        CreateStoragePathSource(source_config, router.get(), &source));

    // Connect the adapters to the manager, and wait for the models to load.
    TF_RETURN_IF_ERROR(ConnectAdaptersToManagerAndAwaitModelLoads(&adapters));

    // Stow the source components.
    storage_path_source_and_router_ = {source.get(), router.get()};
    manager_.AddDependency(std::move(source));
    manager_.AddDependency(std::move(router));
    for (auto& entry : adapters.platform_adapters) {
      auto& adapter = entry.second;
      manager_.AddDependency(std::move(adapter));
    }
    manager_.AddDependency(std::move(adapters.error_adapter));
  } else {
    // Create a fresh servable state monitor, to avoid getting confused if we're
    // re-loading a model-version that has previously been unloaded.
    ServableStateMonitor fresh_servable_state_monitor(
        servable_event_bus_.get());

    // Figure out which models are new.
    const std::set<string> new_models = NewModelNamesInSourceConfig(
        storage_path_source_and_router_->source->config(), source_config);

    // Now we're ready to start reconfiguring the elements of the Source->
    // Manager pipeline ...

    // First, add the new routes without removing the old ones.
    DynamicSourceRouter<StoragePath>::Routes old_and_new_routes;
    const Status union_status =
        UnionRoutes(storage_path_source_and_router_->router->GetRoutes(),
                    routes, &old_and_new_routes);
    if (!union_status.ok()) {
      // ValidateNoModelsChangePlatforms() should have detected any conflict.
      DCHECK(false);
      return errors::Internal("Old and new routes conflict.");
    }
    TF_RETURN_IF_ERROR(ReloadRoutes(old_and_new_routes));

    // Change the source config. Among other things this will cause it to emit
    // tear-downs of any models that aren't present in the new config.
    TF_RETURN_IF_ERROR(ReloadStoragePathSourceConfig(source_config));

    // Now that any old models are out of the picture, remove the old routes.
    TF_RETURN_IF_ERROR(ReloadRoutes(routes));

    // Wait for any new models to get loaded and become available.
    TF_RETURN_IF_ERROR(
        WaitUntilModelsAvailable(new_models, &fresh_servable_state_monitor));
  }
  return Status::OK();
}
```
实际建立的关系是Source->Router->Adapter->AspiredVersionsManager(Target)。Source的实现者是FileSystemStoragePathSource，该对象会在文件目录内容变化时通知观察者。

在AspiredVersionsManager内部实际是AspiredVersionsManagerTargetImpl来实现的Target。AspiredVersionsManagerTargetImpl继承TargetBase而来。实际相当于AspiredVersionsManager实现的时候将接口放到了AspiredVersionsManagerTargetImpl内部类来实现。

再看下具体的Source，服务启动时候实际是有两个Source实现被创建。

1. class StaticStoragePathSource : public Source<StoragePath>
2. class FileSystemStoragePathSource : public Source<StoragePath>

其中FileSystemStoragePathSource是来实现模型热更新的，有个单独的现成来检测初始化时期的存储路径的文件的变化，在文件变化时，使用Source->Router->Adapter->AspiredVersionsManager(Target)过程来更新模型。