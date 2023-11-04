#include "bertbase.h"
#include "bertencoder.h"
#include "ggml.h"

#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/network/Server.hpp"

#include <iostream>
#include <sstream>

#include <string>
#include <vector>
#include <cstring>

#ifdef WIN32
#include "winsock2.h"
#include "include_win/unistd.h"
typedef int socklen_t;
#define read _read
#define close _close

#define SOCKET_HANDLE SOCKET
#else
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#define SOCKET_HANDLE int
#endif

#define LABEL_COUNT 3

#include OATPP_CODEGEN_BEGIN(DTO)

class ResultDto : public oatpp::DTO {
  DTO_INIT(ResultDto, DTO)

  DTO_FIELD(String, result);
};

#include OATPP_CODEGEN_END(DTO)

#include OATPP_CODEGEN_BEGIN(DTO)
class BertReq : public oatpp::DTO {
    DTO_INIT(BertReq, DTO)

    DTO_FIELD(String, input, "input");
};

#include OATPP_CODEGEN_END(DTO)


#include OATPP_CODEGEN_BEGIN(ApiController) ///< Begin ApiController codegen section

class BertClsController : public oatpp::web::server::api::ApiController {
public:
    BertClsController(std::shared_ptr<ObjectMapper> objectMapper)
        : oatpp::web::server::api::ApiController(objectMapper) 
    {}

    static std::shared_ptr<BertClsController> createShared(
        std::shared_ptr<ObjectMapper> objectMapper // Inject objectMapper component here as default parameter
    ){
        return std::make_shared<BertClsController>(objectMapper);
    }

    ENDPOINT_INFO(classify) {
        info->summary = "classify";

        info->addConsumes<Object<BertReq>>("application/json");

        info->addResponse<Object<ResultDto>>(Status::CODE_200, "application/json");
    }

    ENDPOINT("POST", "classify", classify,
        BODY_DTO(Object<BertReq>, req)) {
        oatpp::Any any = req;
        auto dto = any.retrieve<oatpp::Object<BertReq>>();

        std::vector<float> labels = std::vector<float>(LABEL_COUNT);
        bert::bert_encode_classify(_bctx, _params->n_threads, (*(dto.get()->input)).c_str (), labels.data());

        std::ostringstream stmp;
        stmp<<"[";
        int i = 0;
        for (auto& label : labels) {
            if (i > 0) {
                stmp<<",";
            }
            i++;
            stmp<<label;
        }
        stmp<<"]";

        auto result = ResultDto::createShared();
        result->result = stmp.str();

        return createDtoResponse(Status::CODE_200, result);
    }

    void setBertBaseCtx(bert::BertBaseCtx* ctx_, bert::BertParams *params_) { _bctx = ctx_; _params = params_; }

private:
    bert::BertBaseCtx *_bctx;
    bert::BertParams *_params;
};

#include OATPP_CODEGEN_END(ApiController) ///< End ApiController codegen section


void run(bert::BertBaseCtx *_bctx, bert::BertParams *_params) {
    auto router = oatpp::web::server::HttpRouter::createShared();
    
    auto objectMapper = oatpp::parser::json::mapping::ObjectMapper::createShared();
    objectMapper->getDeserializer()->getConfig()->allowUnknownFields = false;

  std::shared_ptr<BertClsController> myCtrl = BertClsController::createShared(objectMapper);
  myCtrl->setBertBaseCtx(_bctx, _params);

  router->addController(myCtrl);

  auto connectionHandler = oatpp::web::server::HttpConnectionHandler::createShared(router);
  auto connectionProvider = oatpp::network::tcp::server::ConnectionProvider::createShared({"localhost", _params->port, oatpp::network::Address::IP_4});

  /* create server */
  oatpp::network::Server server(connectionProvider, connectionHandler);
  
  OATPP_LOGI("MyApp", "Server running on port %s", connectionProvider->getProperty("port").getData());
  
  server.run();
}


int main(int argc, char ** argv) {
    bert::BertParams params;

    if (bert::bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    bert::BertBaseCtx *bctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if ((bctx = bert::bertencoder_load_from_file(params.model)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }
    oatpp::base::Environment::init();
    run(bctx, &params);
    oatpp::base::Environment::destroy();
    return 0;
}
