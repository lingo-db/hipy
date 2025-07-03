#ifndef BUILTIN_H
#define BUILTIN_H

#include "builtin_commons.h"
#include "builtin_date.h"
#include<iostream>
#include<iomanip>
#include <sstream>
#include <algorithm>
#include <fstream>
#include<unordered_map>

#include "datastructures.h"

namespace {
    // taken from
    // https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string
    inline std::string_view ltrim(std::string_view str) {
        const auto pos(str.find_first_not_of(" \t\n\r\f\v"));
        str.remove_prefix(std::min(pos, str.length()));
        return str;
    }

    inline std::string_view rtrim(std::string_view str) {
        const auto pos(str.find_last_not_of(" \t\n\r\f\v"));
        str.remove_suffix(std::min(str.length() - pos - 1, str.length()));
        return str;
    }

    inline std::string_view trim(std::string_view str) {
        str = ltrim(str);
        str = rtrim(str);
        return str;
    }
}

namespace builtin {
    void print(std::string arg) {
        std::cout << arg << std::endl;
    }

    template<class X>
    inline std::string float_to_string(X arg) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(6) << arg;
        auto result = ss.str();
        result.erase(result.find_last_not_of('0') + 1, std::string::npos);
        if (result.back() == '.') {
            result.push_back('0');
        }
        return result;
    }

    namespace string {
        inline std::string strip(const std::string& arg) {
            return std::string(trim(arg));
        }
        inline std::string rstrip(const std::string& arg) {
            return std::string(rtrim(arg));
        }


        inline std::shared_ptr<std::vector<std::string>> split(const std::string& s, const std::string& pattern,int64_t maxsplit) {
            std::vector<std::string> result;
            size_t pos = 0;
            while (pos != std::string::npos) {
                if(result.size()==maxsplit) {
                    result.push_back(s.substr(pos));
                    break;
                }
                auto foundPos = s.find(pattern, pos);
                if (foundPos == std::string::npos) {
                    result.push_back(s.substr(pos));
                    break;
                }
                result.push_back(s.substr(pos, foundPos - pos));
                pos = foundPos + pattern.size();
            }
            return std::make_shared<std::vector<std::string>>(std::move(result));
        }

        inline std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return s;
        }

        inline std::string upper(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), ::toupper);
            return s;
        }

        inline std::string substr(const std::string& s, int64_t start, int64_t end) {
            start = std::min(start, (int64_t) s.size());
            return std::string(s.substr(start, end - start));
        }

        inline bool contains(const std::string& s, std::string pattern) {
            return s.find(pattern) != std::string::npos;
        }

        inline std::string at(const std::string& s, int64_t pos) {
            return std::string(1, s.at(pos));
        }

        inline int64_t find(const std::string& str, const std::string& sub, int64_t start, int64_t end) {
            auto pos = str.find(sub, start);
            if (pos == std::string::npos || pos + sub.size() > end) {
                return -1;
            } else {
                return pos;
            }
        }

        inline int64_t rfind(const std::string& str, const std::string& sub, int64_t start, int64_t end) {
            end -= sub.size();
            if (end < 0) end = 0;
            auto pos = str.rfind(sub, end);
            if (pos == std::string::npos || pos < start) {
                return -1;
            } else {
                return pos;
            }
        }

        inline std::string replace(const std::string& str, const std::string& oldVal, const std::string& newVal) {
            auto len = oldVal.size();
            std::string output = str;
            size_t pos = output.find(oldVal);
            while (pos != std::string::npos) {
                output.replace(pos, len, newVal);
                pos = output.find(oldVal);
            }
            return output;
        }
    }
    std::shared_ptr<std::vector<std::string>> read_file_to_lines(const std::string& filename) {
        // Open the file
        std::ifstream file(filename);

        // Check if the file is opened successfully
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return nullptr;
        }

        // Create a vector to store lines
        auto lines = std::make_shared<std::vector<std::string>>();

        // Read file line by line
        std::string line;
        while (std::getline(file, line)) {
            lines->push_back(line);
        }

        // Close the file
        file.close();

        return lines;
    }
}
namespace std {
    template<typename... T>
    struct hash<std::tuple<T...>> {
        size_t operator()(const std::tuple<T...>& t) const {
            return hash_combine(t);
        }

    private:
        // Hash combining function for tuple elements
        template<std::size_t I = 0, typename... Ts>
        static typename std::enable_if<I == sizeof...(Ts), size_t>::type
        hash_combine(const std::tuple<Ts...>& t) {
            return 0;
        }

        template<std::size_t I = 0, typename... Ts>
        static typename std::enable_if<I < sizeof...(Ts), size_t>::type
        hash_combine(const std::tuple<Ts...>& t) {
            size_t seed = std::hash<typename std::tuple_element<I, std::tuple<Ts...>>::type>{}(std::get<I>(t));
            return seed ^ hash_combine<I + 1>(t);
        }
    };
}
#endif //BUILTIN_H
