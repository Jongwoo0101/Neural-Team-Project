/*
1. 설명
sort_performance.cpp
loss_acc_log.txt 파일을 읽어 FINAL ACC와 loss를 기준으로 각각 정렬한 후, 원본 로그 형식 그대로 sort_acc.txt와 sort_loss.txt 두 파일에 기록
:cout에 대해서, 한글로 출력시 깨지는 문제가 있어 모두 영어로 구성
2-1. 실행
F6을 눌러도 정상 실행되도록 파일 위치를 세팅했다.
이전의 방식:
실행시 /output폴더로 들어간다. 따라서 src\test>위치에서 "sort_performance.exe"를 실행해야 장상적으로 작동한다.
cd ..
output\"sort_performance.exe"
2-2. 코드 수정: 코드 수정시 2번 방법으로 실행시 src\test>위치에서 아래 코드로 .exe 파일을 갱신해야한다.
g++ sort_performance.cpp -o sort_performance.exe
또는 가장 확실한 방법 중 하나로 실행시에 생성되는 code.exe파일을 제거하고 다시 실행하는 것
4. bits/stdc++.h 헤더파일을 사용하지 않는 경우의 사용해야하는 include헤더파일
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <functional>
5. 변경 사항
find_min_performance.cpp에서 sort_performance.cpp으로 이름이 변경되었다.
모델이 잘 훈련되었는지에 대한 지표로써 loss와 accuracy의 사용이 확정됨에 따라 다음과 같이 변경한다.
output_log.txt에서 loss_acc_log.txt으로 이름 변경
output.bat에서 loss.bat으로 이름 변경
6. 수정 요구사항
*/
#include <bits/stdc++.h>
using namespace std;

// 로그 한 줄의 데이터를 저장할 구조체
struct LogEntry
{
    string original_line;
    double loss;
    double acc;
    // 백분율 분석을 위한 하이퍼파라미터 추출
    string optimizer;
    string lr;
    int batch;
    int iters;
    int depth;
    string act_init;
    double l2;
};

/**
 * @brief 주어진 로그 라인에서 FINAL LOSS와 Test ACC 값을 파싱한다.
 * @param line 로그 한 줄 문자열
 * @return LogEntry 구조체
 */
LogEntry parse_log_line(const string &line)
{
    LogEntry entry;
    entry.original_line = line;

    // --- 1. 하이퍼파라미터 추출 ---
    // 파이썬 포맷: Key: Value,  또는 Key: Value

    // Optimizer (Adam, AdaGrad 등)
    size_t opt_start = line.find("Optimizer: ") + 11;
    size_t opt_end = line.find(',', opt_start);
    entry.optimizer = line.substr(opt_start, opt_end - opt_start);
    entry.optimizer.erase(0, entry.optimizer.find_first_not_of(' ')); // 앞 공백 제거
    entry.optimizer.erase(entry.optimizer.find_last_not_of(' ') + 1); // 뒤 공백 제거

    // LR (0.1, 0.001 등)
    size_t lr_start = line.find("LR: ") + 4;
    size_t lr_end = line.find(',', lr_start);
    entry.lr = line.substr(lr_start, lr_end - lr_start);
    entry.lr.erase(0, entry.lr.find_first_not_of(' '));
    entry.lr.erase(entry.lr.find_last_not_of(' ') + 1);

    // Batch (128, 256 등)
    size_t batch_start = line.find("Batch: ") + 7;
    size_t batch_end = line.find(',', batch_start);
    string batch_str = line.substr(batch_start, batch_end - batch_start);
    batch_str.erase(0, batch_str.find_first_not_of(' '));
    entry.batch = stoi(batch_str); // <--- stoi 사용

    // Iters (500, 1000 등)
    size_t iters_start = line.find("Iters: ") + 7;
    size_t iters_end = line.find(',', iters_start);
    string iters_str = line.substr(iters_start, iters_end - iters_start);
    iters_str.erase(0, iters_str.find_first_not_of(' '));
    entry.iters = stoi(iters_str); // <--- stoi 사용

    // Depth (1, 6 등)
    size_t depth_start = line.find("Depth: ") + 7;
    size_t depth_end = line.find(',', depth_start);
    string depth_str = line.substr(depth_start, depth_end - depth_start);
    depth_str.erase(0, depth_str.find_first_not_of(' '));
    entry.depth = stoi(depth_str); // <--- stoi 사용

    // Act/Init (relu/relu, sigmoid/sigmoid)
    size_t act_start = line.find("Act/Init: ") + 10;
    size_t act_end = line.find(',', act_start);
    entry.act_init = line.substr(act_start, act_end - act_start);
    entry.act_init.erase(0, entry.act_init.find_first_not_of(' '));
    entry.act_init.erase(entry.act_init.find_last_not_of(' ') + 1);

    // L2 (0, 1e-08 등)
    size_t l2_start = line.find("L2: ") + 4;
    size_t l2_end = line.find('|', l2_start); // L2는 파이썬 코드에서 쉼표가 아닌 '|' 바로 앞까지 이어짐
    string l2_str = line.substr(l2_start, l2_end - l2_start);
    l2_str.erase(0, l2_str.find_first_not_of(' '));
    l2_str.erase(l2_str.find_last_not_of(' ') + 1);
    entry.l2 = stod(l2_str);

    // --- 2. 결과 추출 ---

    // FINAL LOSS
    size_t loss_pos = line.find("FINAL LOSS:") + 12;
    size_t loss_end = line.find('|', loss_pos);
    entry.loss = stod(line.substr(loss_pos, loss_end - loss_pos));

    // Test ACC
    size_t acc_pos = line.find("Test ACC:") + 10;
    size_t acc_end = line.find('|', acc_pos); // 마지막 필드이므로 '|'까지
    if (acc_end == string::npos)
        acc_end = line.length();
    entry.acc = stod(line.substr(acc_pos, acc_end - acc_pos));

    return entry;
}

/**
 * @brief 정확도 내림차순 비교 함수
 */
bool compareByAcc(const LogEntry &a, const LogEntry &b)
{
    return a.acc > b.acc;
}

/**
 * @brief 손실 내림차순 비교 함수 (손실은 오름차순이 좋으므로, 작은 값이 앞에 오도록)
 */
bool compareByLoss(const LogEntry &a, const LogEntry &b)
{
    return a.loss < b.loss;
}

/**
 * @brief 상위 N% 데이터의 하이퍼파라미터 분포를 분석하고 포맷팅.
 * @param sorted_data 분석할 데이터 (정확도 내림차순 또는 손실 오름차순)
 * @param total_count 전체 데이터 개수
 * @param percentiles 분석할 백분율 목록
 * @param analysis_standard 분석 기준 (예: "Test ACC" 또는 "FINAL LOSS")
 * @param ofs 결과를 기록할 출력 파일 스트림
 */
void analyze_top_percentiles(const vector<LogEntry> &sorted_data,
                             size_t total_count,
                             const vector<int> &percentiles,
                             const string &analysis_standard,
                             ofstream &ofs)
{

    // --- 헤더 출력 ---
    ofs << "=========================================================================" << "\n";
    ofs << "== Hyperparameter Occurrence Frequency Analysis (Standard: " << analysis_standard << ") ==" << "\n";
    ofs << "=========================================================================" << "\n";
    // 분석 로직은 이전 코드와 동일하게 유지
    vector<pair<string, function<string(const LogEntry &)>>> hparam_getters = {
        {"iters", [](const LogEntry &e)
         { return to_string(e.iters); }},
        {"Opt", [](const LogEntry &e)
         { return e.optimizer; }},
        {"LR", [](const LogEntry &e)
         {
             stringstream ss;
             ss << fixed << setprecision(4) << stod(e.lr);
             return ss.str();
         }},
        {"Depth", [](const LogEntry &e)
         { return to_string(e.depth); }},
        {"Act/Init", [](const LogEntry &e)
         { return e.act_init; }},
        {"L2", [](const LogEntry &e)
         {
             stringstream ss;
             // COMMON_HPARAMS에 1e-8, 1e-4 등이 있으므로 정밀도를 8자리로 설정
             ss << fixed << setprecision(8) << e.l2;
             return ss.str();
         }},
    };

    map<string, map<string, vector<int>>> analysis_results;
    for (const auto &[name, getter] : hparam_getters)
    {
        analysis_results[name] = {};
    }

    // 각 백분율별로 데이터 수집 및 집계
    for (size_t p_idx = 0; p_idx < percentiles.size(); ++p_idx)
    {
        int p = percentiles[p_idx];
        int top_k = static_cast<int>(ceil(total_count * p / 100.0));
        top_k = min(top_k, (int)sorted_data.size());

        for (int i = 0; i < top_k; ++i)
        {
            const LogEntry &entry = sorted_data[i];
            for (const auto &[name, getter] : hparam_getters)
            {
                string value = getter(entry);
                if (analysis_results[name].find(value) == analysis_results[name].end())
                {
                    analysis_results[name][value].resize(percentiles.size(), 0);
                }
                analysis_results[name][value][p_idx]++;
            }
        }
    }

    // --- 결과 포맷 출력 ---
    ofs << "Top Percentiles: {";
    for (size_t i = 0; i < percentiles.size(); ++i)
    {
        ofs << percentiles[i] << "%" << (i < percentiles.size() - 1 ? ", " : "");
    }
    ofs << "}\n";
    ofs << "Total Entries: " << total_count << "\n";
    ofs << "---\n";

    // 분석 결과 출력
    for (const auto &[hparam_name, hparam_values] : analysis_results)
    {
        ofs << hparam_name << ": ";

        for (const auto &[value, counts] : hparam_values)
        {
            ofs << "[" << value << ": {";
            for (size_t p_idx = 0; p_idx < percentiles.size(); ++p_idx)
            {
                int p = percentiles[p_idx];
                int top_k = static_cast<int>(ceil(total_count * p / 100.0));
                top_k = max(1, top_k);

                double percentage = (double)counts[p_idx] / top_k * 100.0;

                ofs << fixed << setprecision(1) << percentage << "%" << (p_idx < percentiles.size() - 1 ? ", " : "");
            }
            ofs << "}] ";
        }
        ofs << "\n";
    }
    ofs << "\n"; // 두 섹션 사이 공백 추가
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // hyperparameter_tuning_loss.py의 결과
    const string input_filename = "../loss_acc_log.txt";
    // test/sort_performance.cpp에서 ../log/ 디렉토리 안에 파일을 생성
    const string output_dir = "../../log/";
    const string acc_output_filename = output_dir + "sort_acc.txt";
    const string loss_output_filename = output_dir + "sort_loss.txt";
    const string percentiles_output_filename = output_dir + "sort_percentiles.txt";

    // 1. 파일 읽기
    ifstream ifs(input_filename);
    if (!ifs.is_open())
    {
        cerr << "Error: Unable to open input file. Check path: " << input_filename << '\n';
        return 1;
    }

    vector<LogEntry> data;
    string line;
    while (getline(ifs, line))
    {
        if (!line.empty())
        {
            // (안전한 parse_log_line 함수를 호출하여 데이터를 LogEntry 벡터에 추가)
            try
            {
                data.push_back(parse_log_line(line));
            }
            catch (const exception &e)
            {
                cerr << "Error parsing line: " << line << " | Reason: " << e.what() << '\n';
                continue;
            }
        }
    }
    ifs.close();

    if (data.empty())
    {
        cout << input_filename << ":No data found or all data failed to parse.\n";
        return 0;
    }

    // total_count는 main 함수 내에서 정의하고 analyze_top_percentiles로 전달해야 함
    size_t total_count = data.size();
    // 상위 1,3,6,12%를 잡기 위한 값
    vector<int> percentiles = {1, 3, 6, 12};
    // 2. 데이터 정렬
    // ACC 내림차순 정렬 (높은 ACC가 상위)
    vector<LogEntry> acc_sorted_data = data;
    sort(acc_sorted_data.begin(), acc_sorted_data.end(), compareByAcc);

    // LOSS 오름차순 정렬 (낮은 LOSS가 상위)
    vector<LogEntry> loss_sorted_data = data;
    sort(loss_sorted_data.begin(), loss_sorted_data.end(), compareByLoss);

    // 3. 정렬된 파일 기록 (sort_acc.txt, sort_loss.txt)

    // sort_acc.txt 기록
    ofstream acc_ofs(acc_output_filename);
    if (acc_ofs.is_open())
    {
        for (const auto &entry : acc_sorted_data)
        {
            acc_ofs << entry.original_line << "\n";
        }
        acc_ofs.close();
        cout << "Successfully created " << acc_output_filename << '\n';
    }
    else
    {
        cerr << "Error: Unable to open " << acc_output_filename << ". Check if the directory exists." << '\n';
    }

    // sort_loss.txt 기록
    ofstream loss_ofs(loss_output_filename);
    if (loss_ofs.is_open())
    {
        for (const auto &entry : loss_sorted_data)
        {
            loss_ofs << entry.original_line << "\n";
        }
        loss_ofs.close();
        cout << "Successfully created " << loss_output_filename << '\n';
    }
    else
    {
        cerr << "Error: Unable to open " << loss_output_filename << ". Check if the directory exists.\n";
    }

    // 4. 백분율 분석 및 기록 (sort_percentiles.txt)
    // ofstream을 열 때 std::ios::out 플래그가 기본값이므로 파일이 새로 생성되거나 덮어쓰기 됩니다.
    ofstream percentiles_ofs(percentiles_output_filename);
    if (!percentiles_ofs.is_open())
    {
        cerr << "Error: Unable to open " << percentiles_output_filename << ". Check if the directory exists: " << output_dir << '\n';
        return 1;
    }
    // 4-1. ACC 기준 분석 수행 (ACC 내림차순 데이터 사용)
    analyze_top_percentiles(acc_sorted_data, total_count, percentiles, "Test ACC", percentiles_ofs);

    // 4-2. LOSS 기준 분석 수행 (LOSS 오름차순 데이터 사용)
    analyze_top_percentiles(loss_sorted_data, total_count, percentiles, "FINAL LOSS", percentiles_ofs);

    percentiles_ofs.close();
    cout << "Successfully created " << percentiles_output_filename << '\n';
}